import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.train import treinar_gridsearch, treinar_optuna
from src.utils import carregar_dados_dengue_de_release

#==============================INPUT==================================================
# Configurações gerais
init = {
    'project_name': 'Assistente_Diagnostico_Dengue',
    'tags': ['teste'],
    'name': 'Dengue_v1',
    'df_name': 'df_dengue_reduzido_20'
}

# Seleciona experimento
yaml_file = "xgb_grid_1"
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml"

# Define o target (NÃO alterar)
target = 'RISCO_GRAVIDADE_grave'

# Seleciona quais métodos de seleção de hiperparãmteros usar
use_gridsearch = True
use_optuna = False

#=====================================================================================


# Carrega config YAML
with open(file_path, "r") as f:
    config = yaml.safe_load(f)

# Carrega df reduzido
df_dengue_reduzido = pd.read_csv('Treinamentos/data/df_dengue_reduzido_20.csv')

# VERIFICAR MELHOR SCORING PARA APLICA_PARAMETROS EM MODELS
# REVER PLOTS DE UTILS

# Roda os experimentos
if use_gridsearch:
    # Roda experimentos com GridSearch
    _, _, _ = treinar_gridsearch(df_dengue_reduzido, target, config, init)

if use_optuna:
    # Roda experimentos com Optuna
    _, _, _ = treinar_optuna(df_dengue_reduzido, target, config, init)
