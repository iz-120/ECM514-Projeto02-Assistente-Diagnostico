import pandas as pd
import numpy as np

def criar_df_dict(df, top_produtos, target='QUANTIDADE_M'):
    df_dict = {}
    for produto in top_produtos:
        # Filtra o produto
        df_produto = df[df['PREFIXO'] == produto].copy()
        # Remove coluna de prefixo
        df_produto.drop(columns=['PREFIXO'], inplace=True)
        # Ordena por data
        df_produto['DATA'] = pd.to_datetime(df_produto['DATA'])
        df_produto = df_produto.sort_values(by='DATA').reset_index(drop=True)
        # Faz resample mensal
        df_produto = df_produto.set_index('DATA').resample('ME').sum().reset_index()
        # Colunas de data (mês e ano)
        df_produto['MES'] = df_produto['DATA'].dt.month
        df_produto['ANO'] = df_produto['DATA'].dt.year
        # Amazena e remove coluna de data
        datas = df_produto['DATA'].copy()
        df_produto.drop(columns=['DATA'], inplace=True)
        # Estação do ano
        df_produto['ESTACAO'] = pd.cut(df_produto['MES'], 
                                    bins=[0, 2, 5, 8, 11, 12], 
                                    labels=['VERAO', 'OUTONO', 'INVERNO', 'PRIMAVERA', 'VERAO'], 
                                    right=False,
                                    ordered=False)
        df_produto = pd.get_dummies(df_produto, columns=['ESTACAO'], drop_first=True)
        # Para ter circularidade no tempo
        df_produto['SIN_MES'] = np.sin(2 * np.pi * df_produto['MES'] / 12)
        df_produto['COS_MES'] = np.cos(2 * np.pi * df_produto['MES'] / 12)
        df_produto.drop(columns=['MES'], inplace=True)
        # Coluna de média móvel de 2 meses
        df_produto['MEDIA_MOVEL_2_MESES'] = df_produto[target].rolling(window=2, min_periods=1).mean()
        # Coluna de quantidade vendida no mês anterior (lag 1)
        df_produto['QUANTIDADE_MES_ANTERIOR'] = df_produto[target].shift(1)
        # Coluna de quantidade vendida 2 meses atrás (lag 2)
        df_produto['QUANTIDADE_2_MESES_ATRAS'] = df_produto[target].shift(2)
        # Coluna de quantidade vendida no mesmo mês do ano anterior
        df_produto['QUANTIDADE_MES_ANO_ANTERIOR'] = df_produto[target].shift(12)
        # Adiciona ao dicionário
        df_dict[produto] = [df_produto, datas]
    return df_dict

