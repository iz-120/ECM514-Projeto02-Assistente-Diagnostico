import pandas as pd
import yaml
from src.train import treinar_gridsearch, treinar_optuna
from src.nn_model import treinar_nn

#==============================INPUT==================================================
# Configurações gerais
init = {
    'project_name': 'Assistente_Diagnostico_Dengue',
    'tags': ['teste', '1%'],
    'name': 'Dengue_v5',
    'df_name': 'df_dengue_reduzido_1'
}

# Seleciona experimento
yaml_file = "nn_mlp_2"
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml"

# Define o target (NÃO alterar)
target = 'RISCO_GRAVIDADE_grave'

# Seleciona quais métodos de seleção de hiperparâmteros usar
use_gridsearch = False
use_optuna = False
use_nn = True 

#=====================================================================================

# Carrega config YAML
with open(file_path, "r") as f:
    config = yaml.safe_load(f)

# Carrega df reduzido
df_dengue_reduzido = pd.read_csv('Treinamentos/data/'+init['df_name']+'.csv')

# Remove coluna data
if 'DT_NOTIFIC' in df_dengue_reduzido.columns:
    df_dengue_reduzido = df_dengue_reduzido.drop(columns=['DT_NOTIFIC'])

# Roda os experimentos
if use_gridsearch:
    # Roda experimentos com GridSearch
    _, _, _ = treinar_gridsearch(df_dengue_reduzido, target, config, init)

if use_optuna:
    # Roda experimentos com Optuna
    _, _, _ = treinar_optuna(df_dengue_reduzido, target, config, init)

if use_nn:
    # Roda experimento com rede neural (MLP)
    _, _, _ = treinar_nn(df_dengue_reduzido, target, config, init)
