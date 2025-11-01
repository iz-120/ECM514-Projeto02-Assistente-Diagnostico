import pandas as pd
import yaml
from src.train import treinar_gridsearch, treinar_optuna

#==============================INPUT==================================================
# Configurações gerais
init = {
    'project_name': 'Assistente_Diagnostico_Dengue',
    'tags': ['teste'],
    'name': 'Dengue_v2',
    'df_name': 'df_dengue_reduzido_1'
}

# Seleciona experimento
yaml_file = "xgb_grid_2"
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml"

# Define o target (NÃO alterar)
target = 'RISCO_GRAVIDADE_grave'

# Seleciona quais métodos de seleção de hiperparâmteros usar
use_gridsearch = False
use_optuna = True

#=====================================================================================


# Carrega config YAML
with open(file_path, "r") as f:
    config = yaml.safe_load(f)

# Carrega df reduzido
df_dengue_reduzido = pd.read_csv('Treinamentos/data/'+init['df_name']+'.csv')

# Roda os experimentos
if use_gridsearch:
    # Roda experimentos com GridSearch
    _, _, _ = treinar_gridsearch(df_dengue_reduzido, target, config, init)

if use_optuna:
    # Roda experimentos com Optuna
    _, _, _ = treinar_optuna(df_dengue_reduzido, target, config, init)
