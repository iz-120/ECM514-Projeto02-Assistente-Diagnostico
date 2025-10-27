import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wandb

def plotar_real_vs_previsto(datas, y_real, y_previsto, produto_nome, tipo_modelo, tuning, file):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datas, y=y_real, mode='lines+markers', name='Real', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=datas, y=y_previsto, mode='lines+markers', name='Previsto', line=dict(color='red', dash='dot')))
    titulo = f"Real x Previsto - {produto_nome} | Modelo: {tipo_modelo} - {file} - {tuning}"
    fig.update_layout(title=titulo, xaxis_title='Data', yaxis_title='Quantidade vendida', template='plotly_white')
    wandb.log({f"Real_vs_Previsto_{produto_nome}": fig})

def define_train_test(df_produto, target, datas, data_inicio=None, num_meses_previsao=None, config=None):
    """
    Separa treino e teste do dataset, usando datas externas (pois df não tem coluna 'DATA').
    
    Parâmetros
    ----------
    df_produto : pd.DataFrame
        Dataset do produto sem a coluna de datas.
    datas : pd.Series ou pd.DatetimeIndex
        Datas correspondentes às linhas de df_produto.
    data_inicio : str ou pd.Timestamp, opcional
        Data inicial no formato 'YYYY-MM-DD' (último dia do mês).
        Se None, será usado split percentual.
    num_meses_previsao : int, opcional
        Quantidade de meses para previsão (obrigatório se data_inicio for informado).
    config : dict, opcional
        Configuração contendo 'train' -> 'test_size' para fallback percentual.

    Retorna
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """

    # Garante que as datas são datetime
    datas = pd.to_datetime(datas).reset_index(drop=True)
    df_produto = df_produto.reset_index(drop=True)

    # -----------------------
    # MODO 1: SPLIT TEMPORAL
    # -----------------------
    if data_inicio is not None and num_meses_previsao is not None:
        data_inicio = pd.to_datetime(data_inicio)
        data_fim = data_inicio + pd.DateOffset(months=num_meses_previsao-1)

        mask_train = datas < data_inicio
        mask_test = (datas >= data_inicio) & (datas <= data_fim)

        df_train = df_produto[mask_train]
        df_test = df_produto[mask_test]

        datas_test = datas[mask_test]

    # -------------------------
    # MODO 2: SPLIT PERCENTUAL
    # -------------------------
    else:
        if config is None:
            raise ValueError("É necessário fornecer 'config[train][test_size]' para usar split percentual.")

        split = int(len(df_produto) * (1 - config['train']['test_size']))
        df_train, df_test = df_produto.iloc[:split], df_produto.iloc[split:]
        datas_test = datas.iloc[split:]

    # -------------------------
    # Definição de X e y
    # -------------------------
    X_train = df_train.drop(columns=[target]).apply(pd.to_numeric, errors='coerce')
    y_train = df_train[target]

    X_test = df_test.drop(columns=[target]).apply(pd.to_numeric, errors='coerce')
    y_test = df_test[target]

    return X_train, X_test, y_train, y_test, datas_test