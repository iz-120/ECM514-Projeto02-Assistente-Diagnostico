import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wandb
import requests
import zipfile
from io import BytesIO

# ==============================================================================
# CARREGA OS DADOS UTILIZADOS
# ==============================================================================
def carregar_dados_dengue_de_release(url_release: str, ficheiro_csv_no_zip: str) -> pd.DataFrame:
    """
    Descarrega um ficheiro ZIP de uma URL, extrai um CSV específico e o carrega para um DataFrame.

    Args:
        url_release (str): O URL público para o ficheiro .zip na Release do GitHub.
        ficheiro_csv_no_zip (str): O nome exato do ficheiro .csv dentro do .zip.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas contendo os dados do CSV, ou um DataFrame vazio em caso de erro.
    """
    print(f"A descarregar dados de: {url_release}")

    try:
        # Faz o request para obter o conteúdo do ficheiro
        response = requests.get(url_release, timeout=300) # Timeout de 5 minutos
        response.raise_for_status()  # Lança um erro para respostas HTTP > 400

        print("Download concluído com sucesso. A extrair o ficheiro CSV...")

        # Usa BytesIO para tratar o conteúdo do ZIP em memória, sem salvar no disco
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Verifica se o ficheiro CSV esperado está no ZIP
            if ficheiro_csv_no_zip not in z.namelist():
                print(f"ERRO: O ficheiro '{ficheiro_csv_no_zip}' não foi encontrado no ZIP.")
                print(f"Ficheiros disponíveis: {z.namelist()}")
                return pd.DataFrame()

            # Extrai e carrega o CSV para o pandas
            with z.open(ficheiro_csv_no_zip) as f:
                df = pd.read_csv(f, low_memory=False)
                print(f"DataFrame carregado com sucesso! Shape: {df.shape}")
                return df

    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao descarregar o ficheiro. Motivo: {e}")
        return pd.DataFrame()
    except zipfile.BadZipFile:
        print("ERRO: O ficheiro descarregado não é um ZIP válido.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return pd.DataFrame()

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