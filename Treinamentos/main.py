import pandas as pd
import yaml
from src.data_processing import criar_df_dict
from src.train import treinar_produto, treinar_produto_optuna

#==============================INPUT==================================================
# Seleciona experimento
yaml_file = "xgb_grid_1"
file_path = "Modelos/Experimentos/" + yaml_file + ".yaml"

# Define data de início e meses de previsão (opcional)
data_inicio = pd.to_datetime('2024-10-31')
num_meses_previsao = 6
#=====================================================================================


# Carrega config YAML
with open(file_path, "r") as f:
    config = yaml.safe_load(f)

# Carrega dados
df_tipos = pd.read_csv('Modelos/data/df_tipos.csv')
df_nacional_ambar = df_tipos[(df_tipos['TIPO_DEMANDA']=='NACIONAL') & (df_tipos['DESC_COR']=='AMBAR')].copy()
df_nacional_ambar['QUANTIDADE_M'] = df_nacional_ambar['QUANTIDADE'] / 1_000_000

# Seleciona top produtos
data_corte = pd.to_datetime('2024-01-01')
df_top_produtos_periodo = df_nacional_ambar[pd.to_datetime(df_nacional_ambar['DATA']) >= data_corte].groupby('PREFIXO')['QUANTIDADE'].sum().reset_index()
df_top_produtos_periodo = df_top_produtos_periodo.sort_values(by='QUANTIDADE', ascending=False).reset_index(drop=True)
top_produtos_periodo = df_top_produtos_periodo.head(5)['PREFIXO'].tolist()

#==============================INPUT==================================================
# Seleciona apenas as features e o target
features = ['DATA', 'PREFIXO']
target = 'QUANTIDADE'
all = features.copy()
all.append(target)
#=====================================================================================

# Cria df apenas com features e target
df_feat_targ = df_nacional_ambar[all].copy()

# Cria df_dict
df_dict = criar_df_dict(df_feat_targ, top_produtos_periodo)

# Roda experimentos individuais com GridSearch
# for produto, (df_produto, datas) in df_dict.items():
#     _, _, _ = treinar_produto(df_produto, target, datas, produto, config)

# Roda experimentos individuais com Optuna
for produto, (df_produto, datas) in df_dict.items():
    _, _, _ = treinar_produto_optuna(df_produto, target, datas, produto, config, data_inicio, num_meses_previsao)
