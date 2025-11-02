import time
import numpy as np
import pandas as pd
import wandb
import optuna
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight
from src.utils import define_train_test, avaliar_modelo_completo, flatten_config


def criar_modelo_nn(config):
    """
    Cria um MLPClassifier com parâmetros vindos de `config['nn']` ou usando defaults.
    Retorna um sklearn MLPClassifier (não encapsulado em pipeline).
    """
    nn_cfg = config.get('nn', {}) if config is not None else {}
    raw_hls = nn_cfg.get('hidden_layer_sizes', (128, 64))
    # allow formats: tuple/list of ints, list-of-ints, or string like '128-64' or '128,64'
    if isinstance(raw_hls, str):
        parts = raw_hls.replace('-', ',').split(',')
        hidden_layer_sizes = tuple(int(x) for x in parts if x != '')
    elif isinstance(raw_hls, list):
        # could be a flat list [128,64] meaning single architecture, or a list of lists
        if all(isinstance(x, int) for x in raw_hls):
            hidden_layer_sizes = tuple(raw_hls)
        else:
            # fallback: take first if nested list
            first = raw_hls[0]
            hidden_layer_sizes = tuple(first) if isinstance(first, (list, tuple)) else tuple(raw_hls)
    else:
        hidden_layer_sizes = tuple(raw_hls)
    activation = nn_cfg.get('activation', 'relu')
    alpha = nn_cfg.get('alpha', 1e-4)
    batch_size = nn_cfg.get('batch_size', 32)
    max_iter = nn_cfg.get('max_iter', 200)
    early_stopping = nn_cfg.get('early_stopping', True)
    random_state = config['train']['random_state'] if config is not None else 42

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state
    )
    return clf


def treinar_nn(df_dengue, target, config, init):
    """
    Treina uma rede neural simples (MLP) com pipeline de imputação e escalonamento.
    Faz CV estratificado para recall e loga resultados no W&B.

    Retorna: pipeline_treinado, y_pred, None (sem df_import)
    """
    # Inicia run
    wandb.init(project=init['project_name'], tags=init['tags'], config=config, name=init['name'])

    # Flatten config e atualiza wandb.config
    flattened_config = flatten_config(config) if config is not None else {}
    wandb.config.update(flattened_config)

    # Split treino/test
    X_train, X_test, y_train, y_test = define_train_test(df_dengue, target, config=config)

    # remover colunas que são totalmente NaN no conjunto de treino (imputer não consegue operar nelas)
    cols_allna = X_train.columns[X_train.isna().all()].tolist()
    if cols_allna:
        X_train = X_train.drop(columns=cols_allna)
        X_test = X_test.drop(columns=cols_allna, errors='ignore')
        wandb.log({"nn/dropped_allna_columns": cols_allna})

    # calcula class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    wandb.log({"nn/class_weight": class_weight_dict})

    # Pipeline base (MinMaxScaler + SMOTE will be added via imblearn pipeline when needed)
    # Use imbalanced-learn pipeline so SMOTE is applied only on training folds
    from imblearn.over_sampling import SMOTE
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=config['train'].get('random_state', 42))),
        ('clf', criar_modelo_nn(config))
    ])

    # Suporta modos: simple, grid e optuna
    param_format = config.get('model', {}).get('param_format', 'simple').lower() if config is not None else 'simple'
    best_pipeline = None
    best_search_info = {}

    if param_format in ['grid', 'gridsearch', 'gridsearchcv']:
        # Log do método de otimização
        wandb.log({"Otimização": "GridSearchCV"})

        # Constroi param_grid a partir da seção 'nn' (espera listas para grid)
        nn_cfg = config.get('nn', {})
        param_grid = {}
        for k, v in nn_cfg.items():
            key_name = f'clf__{k}'
            if k == 'hidden_layer_sizes':
                converted = []
                # v is expected to be a list of options; each option may be a string '128-64', a list [128,64], or an int
                for item in v if isinstance(v, list) else [v]:
                    if isinstance(item, str):
                        parts = item.replace('-', ',').split(',')
                        converted.append(tuple(int(x) for x in parts if x != ''))
                    elif isinstance(item, (list, tuple)):
                        converted.append(tuple(int(x) for x in item))
                    elif isinstance(item, int):
                        converted.append((int(item),))
                param_grid[key_name] = converted
            else:
                param_grid[key_name] = v

        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=config['train']['cv'],
            scoring=config['cross_val']['scoring'],
            n_jobs=config['cross_val'].get('n_jobs', 1),
            verbose=config['cross_val'].get('verbose', 0),
            return_train_score=True
        )
        # GridSearchCV não propaga sample_weight automaticamente; rodamos sem sample_weight
        gs.fit(X_train, y_train)
        best_pipeline = gs.best_estimator_
        best_search_info = {'method': 'grid', 'best_score': gs.best_score_, 'best_params': gs.best_params_}
        wandb.log({'grid/melhor_cv': gs.best_score_, 'grid/melhor_params': gs.best_params_})

    elif param_format in ['optuna', 'optuna_search']:
        # Log do método de otimização
        wandb.log({"Otimização": "Optuna"})

        nn_cfg = config.get('nn', {})

        def objective(trial):
            trial_params = {}
            for k, v in nn_cfg.items():
                if isinstance(v, list):
                    # let Optuna pick among the provided choices (they may be strings like '128-64')
                    trial_val = trial.suggest_categorical(k, v)
                else:
                    trial_val = v
                trial_params[k] = trial_val

            # Coerce types to those expected by sklearn MLP
            typed = {}
            for k, v in trial_params.items():
                if k == 'hidden_layer_sizes':
                    if isinstance(v, str):
                        parts = v.replace('[', '').replace(']', '').split(',')
                        parts= list(int(x) for x in parts if x != '')
                        typed[k] = tuple(int(x) for x in parts if x != '')
                    elif isinstance(v, (list, tuple)):
                        typed[k] = tuple(int(x) for x in v)
                    else:
                        typed[k] = v
                elif k in ('alpha', 'learning_rate'):
                    typed[k] = float(v)
                elif k in ('batch_size', 'max_iter', 'n_estimators', 'num_leaves', 'max_depth'):
                    typed[k] = int(v)
                elif k == 'early_stopping':
                    if isinstance(v, str):
                        typed[k] = v.lower() in ('true', '1', 'yes')
                    else:
                        typed[k] = bool(v)
                else:
                    typed[k] = v

            model = MLPClassifier(
                hidden_layer_sizes=typed.get('hidden_layer_sizes', (128, 64)),
                activation=typed.get('activation', 'relu'),
                alpha=typed.get('alpha', 1e-4),
                batch_size=typed.get('batch_size', 32),
                max_iter=typed.get('max_iter', 200),
                early_stopping=typed.get('early_stopping', True),
                random_state=config['train']['random_state']
            )

            from imblearn.over_sampling import SMOTE
            temp_pipe = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler()),
                ('smote', SMOTE(random_state=config['train'].get('random_state', 42))),
                ('clf', model)
            ])

            score = cross_val_score(temp_pipe, X_train, y_train, cv=config['train']['cv'], scoring=config['cross_val']['scoring'], n_jobs=1)
            return score.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config['cross_val'].get('n_trials', 20))

        best_params = study.best_params

        if isinstance(best_params["hidden_layer_sizes"], str):
            parts = best_params["hidden_layer_sizes"].replace('[', '').replace(']', '').split(',')
            parts= list(int(x) for x in parts if x != '')
            best_params["hidden_layer_sizes"] = tuple(int(x) for x in parts if x != '')

        best_model = MLPClassifier(
            hidden_layer_sizes=best_params.get('hidden_layer_sizes', (128, 64)),
            activation=best_params.get('activation', 'relu'),
            alpha=best_params.get('alpha', 1e-4),
            batch_size=best_params.get('batch_size', 32),
            max_iter=best_params.get('max_iter', 200),
            early_stopping=best_params.get('early_stopping', True),
            random_state=config['train']['random_state']
        )
        best_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler()),
            ('smote', SMOTE(random_state=config['train'].get('random_state', 42))),
            ('clf', best_model)
        ])
        best_search_info = {'method': 'optuna', 'best_value': study.best_value, 'best_params': study.best_params}
        wandb.log({'optuna/melhor_valor': study.best_value, 'optuna/melhor_params': study.best_params, 'optuna/n_trials': len(study.trials)})

    else:
        best_pipeline = pipeline

    # Cross-validation manual (para poder usar sample_weight)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)


    # Treina final em todo o conjunto de treino usando best_pipeline
    inicio = time.time()
    best_pipeline.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    # Predições
    y_pred = best_pipeline.predict(X_test)
    try:
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_proba = None

    # Avaliação e logging
    _ = avaliar_modelo_completo(
        y_test, y_pred, y_pred_proba,
        nome_modelo=f"MLP - NN"
    )

    wandb.log({
        "info/memoria_fim": None,
        "info/cpu_fim": None,
        "tempo/treino": tempo_treino
    })

    # Registro do dataset no W&B
    path_df = 'Treinamentos/data/'+init['df_name']+'.csv'
    artifact = wandb.Artifact(
        "df_dengue",
        type="dataset",
        description=f"Dataset usado para treinar MLP"
    )
    df_dengue.to_csv(path_df, index=False)
    artifact.add_file(path_df)
    wandb.log_artifact(artifact)

    # Log best search info if present
    if best_search_info:
        wandb.log({'nn/best_search': best_search_info})

    wandb.finish()
    return best_pipeline, y_pred, None
