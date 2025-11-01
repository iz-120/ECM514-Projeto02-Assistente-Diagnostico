import time
import numpy as np
import pandas as pd
import wandb
import optuna
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
    hidden_layer_sizes = tuple(nn_cfg.get('hidden_layer_sizes', (128, 64)))
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

    # calcula class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    wandb.log({"nn/class_weight": class_weight_dict})

    # Pipeline base
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', criar_modelo_nn(config))
    ])

    # Suporta modos: simple, grid e optuna
    param_format = config.get('model', {}).get('param_format', 'simple').lower() if config is not None else 'simple'
    best_pipeline = None
    best_search_info = {}

    if param_format in ['grid', 'gridsearch', 'gridsearchcv']:
        # Constroi param_grid a partir da seção 'nn' (espera listas para grid)
        nn_cfg = config.get('nn', {})
        param_grid = {}
        for k, v in nn_cfg.items():
            key_name = f'clf__{k}'
            if k == 'hidden_layer_sizes' and isinstance(v, list):
                # converter cada opção para tupla
                converted = []
                for item in v:
                    if isinstance(item, list):
                        converted.append(tuple(item))
                    else:
                        # se receberam uma lista plana, transformamos em tupla única
                        converted = [tuple(v)]
                        break
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
        nn_cfg = config.get('nn', {})

        def objective(trial):
            trial_params = {}
            for k, v in nn_cfg.items():
                if isinstance(v, list):
                    if k == 'hidden_layer_sizes':
                        choices = []
                        for item in v:
                            if isinstance(item, list):
                                choices.append(tuple(item))
                            else:
                                choices.append((int(item),))
                        trial_val = trial.suggest_categorical(k, choices)
                    else:
                        trial_val = trial.suggest_categorical(k, v)
                else:
                    trial_val = v
                trial_params[k] = trial_val

            model = MLPClassifier(
                hidden_layer_sizes=trial_params.get('hidden_layer_sizes', (128, 64)),
                activation=trial_params.get('activation', 'relu'),
                alpha=trial_params.get('alpha', 1e-4),
                batch_size=trial_params.get('batch_size', 32),
                max_iter=trial_params.get('max_iter', 200),
                early_stopping=trial_params.get('early_stopping', True),
                random_state=config['train']['random_state']
            )

            temp_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            score = cross_val_score(temp_pipe, X_train, y_train, cv=config['train']['cv'], scoring=config['cross_val']['scoring'], n_jobs=1)
            return score.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config['cross_val'].get('n_trials', 20))

        best_params = study.best_params
        best_model = MLPClassifier(
            hidden_layer_sizes=best_params.get('hidden_layer_sizes', (128, 64)),
            activation=best_params.get('activation', 'relu'),
            alpha=best_params.get('alpha', 1e-4),
            batch_size=best_params.get('batch_size', 32),
            max_iter=best_params.get('max_iter', 200),
            early_stopping=best_params.get('early_stopping', True),
            random_state=config['train']['random_state']
        )
        best_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', best_model)
        ])
        best_search_info = {'method': 'optuna', 'best_value': study.best_value, 'best_params': study.best_params}
        wandb.log({'optuna/melhor_valor': study.best_value, 'optuna/melhor_params': study.best_params, 'optuna/n_trials': len(study.trials)})

    else:
        best_pipeline = pipeline

    # Cross-validation manual (para poder usar sample_weight)
    skf = StratifiedKFold(n_splits=config['train']['cv'] if config is not None else 5, shuffle=True, random_state=config['train']['random_state'] if config is not None else 42)
    recalls = []
    fold = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold += 1
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sample_weight = y_tr.map(lambda c: class_weight_dict[c]).values
        try:
            best_pipeline.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
        except TypeError:
            best_pipeline.fit(X_tr, y_tr)
        y_pred = best_pipeline.predict(X_val)
        r = recall_score(y_val, y_pred)
        recalls.append(r)
        wandb.log({f"cv/fold_{fold}_recall": r})

    mean_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    wandb.log({"cv/mean_recall": mean_recall})

    # Treina final em todo o conjunto de treino usando best_pipeline
    inicio = time.time()
    sample_weight_full = y_train.map(lambda c: class_weight_dict[c]).values
    try:
        best_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_full)
    except TypeError:
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
        "tempo/treino": tempo_treino,
        "nn/mean_cv_recall": mean_recall
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
