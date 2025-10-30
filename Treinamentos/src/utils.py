import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wandb

# ==============================================================================
# PLOTA GRÁFICOS DE REAL VC PREVISTO
# ==============================================================================
def plotar_real_vs_previsto(datas, y_real, y_previsto, produto_nome, tipo_modelo, tuning, file):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datas, y=y_real, mode='lines+markers', name='Real', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=datas, y=y_previsto, mode='lines+markers', name='Previsto', line=dict(color='red', dash='dot')))
    titulo = f"Real x Previsto - {produto_nome} | Modelo: {tipo_modelo} - {file} - {tuning}"
    fig.update_layout(title=titulo, xaxis_title='Data', yaxis_title='Quantidade vendida', template='plotly_white')
    wandb.log({f"Real_vs_Previsto_{produto_nome}": fig})

# ==============================================================================
# DEFINE TREINO E TESTE
# ==============================================================================
def define_train_test(df_dengue, target, config):
    """
    Separa treino e teste do dataset, usando datas externas (pois df não tem coluna 'DATA').
    
    Parâmetros
    ----------
    df_produto : pd.DataFrame
        Dataset do produto sem a coluna de datas.
    percent_split : dict, opcional
        Configuração contendo 'train' -> 'test_size' para fallback percentual.

    Retorna
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    # SPLIT PERCENTUAL
    if config is None:
        raise ValueError("É necessário fornecer 'config[train][test_size]' para usar split percentual.")

    split = int(len(df_dengue) * (1 - config['train']['test_size']))
    df_train, df_test = df_dengue.iloc[:split], df_dengue.iloc[split:]

    # -------------------------
    # Definição de X e y
    # -------------------------
    X_train = df_train.drop(columns=[target]).apply(pd.to_numeric, errors='coerce')
    y_train = df_train[target]

    X_test = df_test.drop(columns=[target]).apply(pd.to_numeric, errors='coerce')
    y_test = df_test[target]

    return X_train, X_test, y_train, y_test