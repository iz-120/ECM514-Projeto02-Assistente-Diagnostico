import pandas as pd
import yaml
from src.train import treinar_gridsearch, treinar_optuna
from src.nn_model import treinar_nn

#==============================LOGIN WANDB============================================

# Executar essa linha apenas na primeira execução com o novo login
#wandb.login(key='sua_chave_API')  # Substitua pela sua chave de API do Weights & Biases

#==============================INPUT==================================================
# Configurações gerais
init = {
    'org_name': 'izabel-sampaio-org',
    'project_name': 'Assistente_Diagnostico_Dengue',
    'tags': ['5%'],
    'name': 'Dengue_v6',
    'df_name': 'df_dengue_reduzido_5'
}

# Seleciona experimento
yaml_file = "logreg_4"
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml"

# Define o target (NÃO alterar)
target = 'RISCO_GRAVIDADE_grave'

# Seleciona quais métodos de seleção de hiperparâmteros usar
use_grid_or_randomized_search = False
use_optuna = True
use_nn = False 

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
if use_grid_or_randomized_search:
    # Roda experimentos com GridSearch
    _, _, _ = treinar_gridsearch(df_dengue_reduzido, target, config, init)

if use_optuna:
    # Roda experimentos com Optuna
    _, _, _ = treinar_optuna(df_dengue_reduzido, target, config, init)

if use_nn:
    # Roda experimento com rede neural (MLP)
    _, _, _ = treinar_nn(df_dengue_reduzido, target, config, init)
