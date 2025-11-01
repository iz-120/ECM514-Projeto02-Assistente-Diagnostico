import time
import psutil
import wandb
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score
from src.models import criar_modelo, aplica_parametros
from src.utils import define_train_test, avaliar_modelo_completo
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb

def treinar_gridsearch(df_dengue, target, config, init):
    """
    Treina modelo usando GridSearch para otimização de hiperparâmetros
    """
    # Inicia run
    wandb.init(project=init['project_name'], tags=init['tags'], config=config, name=init['name'])
    
    # Log do método de otimização
    wandb.log({"Otimização": "GridSearchCV"})
    
    # Define grupos de treino e teste
    X_train, X_test, y_train, y_test = define_train_test(df_dengue, target, config=config)
    
    # Monitora recursos
    wandb.log({
        "info/memoria_inicio": psutil.virtual_memory().percent,
        "info/cpu_inicio": psutil.cpu_percent()
    })
    
    # Seleciona e configura o modelo base
    modelo_base = criar_modelo(config)
    modelo = aplica_parametros(modelo_base, config)

    # Treina e mede tempo
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    # Predições
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    # Avaliação usando as funções de utils.py
    _ = avaliar_modelo_completo(
        y_test, y_pred, y_pred_proba,
        nome_modelo=f"{config['model']['type'].upper()} - GridSearch"
    )

    # Log dos resultados da grid
    wandb.log({
        "grid/melhor_cv": modelo.best_score_,
        "grid/n_splits_cv": modelo.n_splits_,
        "grid/melhor_params": modelo.best_params_
    })

    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        modelo.best_estimator_, X_train, y_train,
        cv=config['cross_val']['cv'], scoring=config['cross_val']['scoring'], n_jobs=config['cross_val']['n_jobs']
    )
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(train_scores, axis=1),
        mode='lines+markers',
        name="Treino"
    ))
    fig_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(test_scores, axis=1),
        mode='lines+markers',
        name="Validação"
    ))
    fig_curve.update_layout(
        title=f"Curva de Aprendizado - {init['df_name']}",
        xaxis_title="Nº de amostras",
        yaxis_title="AUC"
    )
    wandb.log({"visualizacoes/curva_aprendizado": fig_curve})

    # Importância das features
    if hasattr(modelo.best_estimator_, 'feature_importances_'):
        importances = modelo.best_estimator_.feature_importances_
        features = X_train.columns
        df_import = pd.DataFrame({
            "Feature": features,
            "Importância": importances
        }).sort_values("Importância", ascending=False)

        fig_imp = go.Figure(go.Bar(
            x=df_import["Importância"],
            y=df_import["Feature"],
            orientation="h"
        ))
        fig_imp.update_layout(title=f"Importância das Variáveis - {init['df_name']}")
        wandb.log({
            "visualizacoes/importancia_features": fig_imp,
            "tabelas/importancia_features": wandb.Table(dataframe=df_import)
        })
    
    # Log final de recursos
    wandb.log({
        "info/memoria_fim": psutil.virtual_memory().percent,
        "info/cpu_fim": psutil.cpu_percent(),
        "tempo/treino": tempo_treino
    })

    # Registro do dataset
    path_df = 'Treinamentos/data/'+init['df_name']+'.csv'
    artifact = wandb.Artifact(
        "df_dengue",
        type="dataset",
        description=f"Dataset usado para treinar {config['model']['type']} com GridSearch"
    )
    df_dengue.to_csv(path_df, index=False)
    artifact.add_file(path_df)
    wandb.log_artifact(artifact)

    wandb.finish()
    return modelo, y_pred, df_import

def treinar_optuna(df_dengue, target, config, init):
    """
    Treina modelo usando Optuna para otimização de hiperparâmetros
    """
    # Inicia run
    wandb.init(project=init['project_name'], tags=init['tags'], config=config, name=init['name'])
    
    # Log do método de otimização
    wandb.log({"Otimização": "Optuna"})
    
    # Define grupos de treino e teste
    X_train, X_test, y_train, y_test = define_train_test(df_dengue, target, config=config)
    
    # Monitora recursos
    wandb.log({
        "info/memoria_inicio": psutil.virtual_memory().percent,
        "info/cpu_inicio": psutil.cpu_percent()
    })

    def objective(trial):
        # Configuração dos parâmetros para otimização
        if config['model']['type'].lower() == "xgboost":
            trial_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", config['model']['params']["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", config['model']['params']["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", config['model']['params']["learning_rate"]),
                "subsample": trial.suggest_categorical("subsample", config['model']['params']["subsample"]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", config['model']['params']["colsample_bytree"]),
                "gamma": trial.suggest_categorical("gamma", config['model']['params']["gamma"]),
                "random_state": config['train']['random_state'],
                "objective": "binary:logistic"  # Alterado para classificação binária
            }
            model = xgb.XGBClassifier(**trial_params)
        
        elif config['model']['type'].lower() == "lightgbm":
            trial_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", config['model']['params']["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", config['model']['params']["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", config['model']['params']["learning_rate"]),
                "num_leaves": trial.suggest_categorical("num_leaves", config['model']['params']["num_leaves"]),
                "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", config['model']['params']["min_data_in_leaf"]),
                "feature_fraction": trial.suggest_categorical("feature_fraction", config['model']['params']["feature_fraction"]),
                "bagging_fraction": trial.suggest_categorical("bagging_fraction", config['model']['params']["bagging_fraction"]),
                "bagging_freq": trial.suggest_categorical("bagging_freq", config['model']['params']["bagging_freq"]),
                "random_state": config['train']['random_state'],
                "objective": "binary"  # Alterado para classificação binária
            }
            model = lgb.LGBMClassifier(**trial_params)

        # Cross-validation
        score = cross_val_score(
            model, X_train, y_train,
            cv=config['train']['cv'], scoring=config['cross_val']['scoring'], n_jobs=config['cross_val']['n_jobs']
        )
        return -score.mean()  # Negativo pois o Optuna minimiza

    # Otimização com Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # Log dos resultados da otimização
    wandb.log({
        "optuna/melhor_valor": -study.best_value,  # Convertendo de volta para positivo
        "optuna/melhor_trial": study.best_trial.number,
        "optuna/n_trials": len(study.trials)
    })

    # Treinar modelo final
    best_params = study.best_params
    if config['model']['type'].lower() == "xgboost":
        modelo = xgb.XGBClassifier(**best_params)
    else:
        modelo = lgb.LGBMClassifier(**best_params)

    # Treina e mede tempo
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    # Predições
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    # Avaliação usando as funções de utils.py
    _ = avaliar_modelo_completo(
        y_test, y_pred, y_pred_proba,
        nome_modelo=f"{config['model']['type'].lower().upper()} - Optuna"
    )

    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        modelo, X_train, y_train,
        cv=config['train']['cv'], scoring=config['cross_val']['scoring'], n_jobs=config['cross_val']['n_jobs']
    )
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(train_scores, axis=1),
        mode='lines+markers',
        name="Treino"
    ))
    fig_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(test_scores, axis=1),
        mode='lines+markers',
        name="Validação"
    ))
    fig_curve.update_layout(
        title=f"Curva de Aprendizado - {init['df_name']}",
        xaxis_title="Nº de amostras",
        yaxis_title="AUC"
    )
    wandb.log({"visualizacoes/curva_aprendizado": fig_curve})

    # Importância das features
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        features = X_train.columns
        df_import = pd.DataFrame({
            "Feature": features,
            "Importância": importances
        }).sort_values("Importância", ascending=False)

        fig_imp = go.Figure(go.Bar(
            x=df_import["Importância"],
            y=df_import["Feature"],
            orientation="h"
        ))
        fig_imp.update_layout(title=f"Importância das Variáveis - {init['df_name']}")
        wandb.log({
            "visualizacoes/importancia_features": fig_imp,
            "tabelas/importancia_features": wandb.Table(dataframe=df_import)
        })
    
    # Log final de recursos e parâmetros
    wandb.log({
        "info/memoria_fim": psutil.virtual_memory().percent,
        "info/cpu_fim": psutil.cpu_percent(),
        "tempo/treino": tempo_treino,
        "optuna/melhores_params": best_params
    })

    # Registro do dataset
    path_df = 'Treinamentos/data/'+init['df_name']+'.csv'
    artifact = wandb.Artifact(
        "df_dengue",
        type="dataset",
        description=f"Dataset usado para treinar {config['model']['type'].lower()} com Optuna"
    )
    df_dengue.to_csv(path_df, index=False)
    artifact.add_file(path_df)
    wandb.log_artifact(artifact)

    wandb.finish()
    return modelo, y_pred, df_import
