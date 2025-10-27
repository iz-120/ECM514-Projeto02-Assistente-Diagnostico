import time
import psutil
import wandb
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from src.models import criar_modelo, aplica_parametros
from src.utils import plotar_real_vs_previsto, define_train_test
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb

def treinar_produto(df_produto, target, datas, produto_nome, config, data_inicio=None, num_meses_previsao=None):
    # Inicia run
    wandb.init(project="Demanda_Farmaceutica_v2", tags=['v2', 'testes'], config=config, name=f"{produto_nome}")
    
    # Define grupos de treino e teste
    if data_inicio is not None and num_meses_previsao is not None:
        X_train, X_test, y_train, y_test, datas_test = define_train_test(df_produto, target, datas, data_inicio=data_inicio, num_meses_previsao=num_meses_previsao)
        # Loga data de início da previsão e número de meses que foram previstos
        wandb.log({
            "Data_Inicio_Previsao": data_inicio,
            "Num_Meses_Previsao": num_meses_previsao
        })
    else:
        X_train, X_test, y_train, y_test, datas_test = define_train_test(df_produto, target, datas, config=config)
        # Loga data de início da previsão e número de meses que foram previstos
        wandb.log({
            "Data_Inicio_Previsao": datas_test['DATA'].iloc[0],
            "Num_Meses_Previsao": len(datas_test)
        })

    
    # Seleciona o modelo
    modelo_base = criar_modelo(config['model']['type'], random_state=config['train']['random_state'])

    # Aplica os parâmetros
    modelo = aplica_parametros(modelo_base, config)

    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    stde = np.std(y_pred - y_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
    smape = 100 * np.mean(2*np.abs(y_pred - y_test)/(np.abs(y_test)+np.abs(y_pred)))

    # Melhor score dos modelos
    best_score = modelo.best_score_

    wandb.log({"MAE": mae, "RMSE": rmse, "MAPE": mape, "wMAPE": wmape, "sMAPE": smape, 
               "R²": r2, "BIAS": bias, "StdE": stde, "Best Score": best_score, "tempo_treino": tempo_treino,
               "CPU (%)": psutil.cpu_percent(), "Memória (%)": psutil.virtual_memory().percent})

    df_resultado = pd.DataFrame({"DATA": datas_test, "QUANTIDADE_REAL": y_test, "QUANTIDADE_PREVISTA": y_pred})
    plotar_real_vs_previsto(df_resultado['DATA'], df_resultado['QUANTIDADE_REAL'], df_resultado['QUANTIDADE_PREVISTA'], produto_nome, config['model']['type'],  config['model']['type'], config['model']['param_format'], config['file'])

    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(modelo.best_estimator_, X_train, y_train, cv=3,
                                                            scoring='neg_mean_absolute_error', n_jobs=-1)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=train_sizes, y=-np.mean(train_scores, axis=1), mode='lines+markers', name="Treino"))
    fig_curve.add_trace(go.Scatter(x=train_sizes, y=-np.mean(test_scores, axis=1), mode='lines+markers', name="Validação"))
    fig_curve.update_layout(title=f"Curva de Aprendizado - {produto_nome}", xaxis_title="Nº de amostras", yaxis_title="MAE")
    wandb.log({f"Curva_Aprendizado_{produto_nome}": fig_curve})

    # Importância das features
    importances = modelo.best_estimator_.feature_importances_
    features = X_train.columns
    df_import = pd.DataFrame({"Feature": features, "Importância": importances}).sort_values("Importância", ascending=False)
    fig_imp = go.Figure(go.Bar(x=df_import["Importância"], y=df_import["Feature"], orientation="h"))
    fig_imp.update_layout(title=f"Importância das Variáveis - {produto_nome}")
    wandb.log({f"Importancia_Variaveis_{produto_nome}": fig_imp})

    # Log dos hiperparâmetros que geraram o melhor modelo (se input for em grid)
    wandb.log(modelo.best_params_)

    # Registro do dataset no W&B sem symlink
    path_df = 'Modelos/data/df_produto.csv'
    artifact = wandb.Artifact("df_produto", type="dataset")
    df_produto.to_csv(path_df, index=False)
    artifact.add_file(path_df)
    wandb.log_artifact(artifact)

    wandb.finish()
    return modelo, df_resultado, df_import

def treinar_produto_optuna(df_produto, target, datas, produto_nome, config, data_inicio=None, num_meses_previsao=None):
    # Inicia run
    wandb.init(project="Demanda_Farmaceutica", tags=['v2', 'testes', 'target_quant'], config=config, name=f"{produto_nome}")
    
    # Define grupos de treino e teste
    if data_inicio is not None and num_meses_previsao is not None:
        X_train, X_test, y_train, y_test, datas_test = define_train_test(df_produto, target, datas, data_inicio=data_inicio, num_meses_previsao=num_meses_previsao)
        # Loga data de início da previsão e número de meses que foram previstos
        wandb.log({
            "Data_Inicio_Previsao": data_inicio,
            "Num_Meses_Previsao": num_meses_previsao
        })
    else:
        X_train, X_test, y_train, y_test, datas_test = define_train_test(df_produto, target, datas, config=config)
        # Loga data de início da previsão e número de meses que foram previstos
        wandb.log({
            "Data_Inicio_Previsao": datas_test.iloc[0],
            "Num_Meses_Previsao": len(datas_test)
        })

    
    # Definição do tipo de modelo
    model_type = config['model']['type'].lower()
    params = config['model']['params']
    cv = config['train']['cv']
    random_state = config['train']['random_state']

    def objective(trial):
        if model_type == "xgboost":
            trial_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", params["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", params["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", params["learning_rate"]),
                "subsample": trial.suggest_categorical("subsample", params["subsample"]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", params["colsample_bytree"]),
                "gamma": trial.suggest_categorical("gamma", params["gamma"]),
                "random_state": random_state,
                "objective": "reg:squarederror"
            }
            model = xgb.XGBRegressor(**trial_params)

        elif model_type == "lightgbm":
            trial_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", params["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", params["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", params["learning_rate"]),
                "num_leaves": trial.suggest_categorical("num_leaves", params["num_leaves"]),
                "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", params["min_data_in_leaf"]),
                "feature_fraction": trial.suggest_categorical("feature_fraction", params["feature_fraction"]),
                "bagging_fraction": trial.suggest_categorical("bagging_fraction", params["bagging_fraction"]),
                "bagging_freq": trial.suggest_categorical("bagging_freq", params["bagging_freq"]),
                "random_state": random_state,
                "objective": "regression"
            }
            model = lgb.LGBMRegressor(**trial_params)

        # Cross-validation
        score = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
        return -score.mean()
    
    # Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    # Treinar modelo final com os melhores parâmetros
    if model_type == "xgboost":
        modelo = xgb.XGBRegressor(**best_params)
    else:
        modelo = lgb.LGBMRegressor(**best_params)

    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    stde = np.std(y_pred - y_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
    smape = 100 * np.mean(2*np.abs(y_pred - y_test)/(np.abs(y_test)+np.abs(y_pred)))

    wandb.log({"MAE": mae, "RMSE": rmse, "MAPE": mape, "wMAPE": wmape, "sMAPE": smape, 
               "R²": r2, "BIAS": bias, "StdE": stde, "Best Score": best_score, "tempo_treino": tempo_treino,
               "CPU (%)": psutil.cpu_percent(), "Memória (%)": psutil.virtual_memory().percent})

    df_resultado = pd.DataFrame({"DATA": datas_test, "QUANTIDADE_REAL": y_test, "QUANTIDADE_PREVISTA": y_pred})
    plotar_real_vs_previsto(df_resultado['DATA'], df_resultado['QUANTIDADE_REAL'], df_resultado['QUANTIDADE_PREVISTA'], produto_nome, config['model']['type'], 'optuna', config['file'])

    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(modelo, X_train, y_train, cv=3,
                                                            scoring='neg_mean_absolute_error', n_jobs=-1)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=train_sizes, y=-np.mean(train_scores, axis=1), mode='lines+markers', name="Treino"))
    fig_curve.add_trace(go.Scatter(x=train_sizes, y=-np.mean(test_scores, axis=1), mode='lines+markers', name="Validação"))
    fig_curve.update_layout(title=f"Curva de Aprendizado - {produto_nome}", xaxis_title="Nº de amostras", yaxis_title="MAE")
    wandb.log({f"Curva_Aprendizado_{produto_nome}": fig_curve})

    # Importância das features
    importances = modelo.feature_importances_
    features = X_train.columns
    df_import = pd.DataFrame({"Feature": features, "Importância": importances}).sort_values("Importância", ascending=False)
    fig_imp = go.Figure(go.Bar(x=df_import["Importância"], y=df_import["Feature"], orientation="h"))
    fig_imp.update_layout(title=f"Importância das Variáveis - {produto_nome}")
    wandb.log({f"Importancia_Variaveis_{produto_nome}": fig_imp})

    # Log dos hiperparâmetros que geraram o melhor modelo (se input for em grid)
    wandb.log(modelo.get_params())
    

    # Registro do dataset no W&B sem symlink
    path_df = 'Modelos/data/df_produto.csv'
    artifact = wandb.Artifact("df_produto", type="dataset")
    df_produto.to_csv(path_df, index=False)
    artifact.add_file(path_df)
    wandb.log_artifact(artifact)

    wandb.finish()
    return modelo, df_resultado, df_import
