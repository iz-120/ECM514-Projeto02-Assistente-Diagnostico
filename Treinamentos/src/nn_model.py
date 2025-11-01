import time
import numpy as np
import pandas as pd
import wandb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
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

    wandb.log({"Otimização": "NN_MLP"})

    # Split treino/test
    X_train, X_test, y_train, y_test = define_train_test(df_dengue, target, config=config)

    # calcula class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    wandb.log({"nn/class_weight": class_weight_dict})

    # Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', criar_modelo_nn(config))
    ])

    # Cross-validation manual (para poder usar sample_weight)
    skf = StratifiedKFold(n_splits=config['train']['cv'] if config is not None else 5, shuffle=True, random_state=config['train']['random_state'] if config is not None else 42)
    recalls = []
    fold = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold += 1
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sample_weight = y_tr.map(lambda c: class_weight_dict[c]).values
        # fit com sample_weight no estimator via nome do step: clf__sample_weight
        pipeline.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
        y_pred = pipeline.predict(X_val)
        r = recall_score(y_val, y_pred)
        recalls.append(r)
        wandb.log({f"cv/fold_{fold}_recall": r})

    mean_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    wandb.log({"cv/mean_recall": mean_recall})

    # Treina final em todo o conjunto de treino
    inicio = time.time()
    sample_weight_full = y_train.map(lambda c: class_weight_dict[c]).values
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_full)
    tempo_treino = time.time() - inicio

    # Predições
    y_pred = pipeline.predict(X_test)
    # probabilidades se disponíveis
    try:
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
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

    wandb.finish()
    return pipeline, y_pred, None
