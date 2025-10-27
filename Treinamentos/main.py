import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
# from src.train import treinar, treinar_optnua
from src.utils import carregar_dados_dengue_de_release

#==============================INPUT==================================================
# Seleciona experimento
yaml_file = "xgb_grid_1"
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml"

# Define o target (NÃO alterar)
target = 'RISCO_GRAVIDADE_grave'

# Seleciona quais métodos de seleção de hiperparãmteros usar
use_gridsearch = False
use_optuna = False

# Para df reduzido
frac = 0.1  # Fração do dataset original
#=====================================================================================


# Carrega config YAML
with open(file_path, "r") as f:
    config = yaml.safe_load(f)

# Carrega dados completos
# URL dos dados a serem recuperados
URL_DADOS_LIMPOS = "https://github.com/iz-120/ECM514-Projeto02-Assistente-Diagnostico/releases/download/v1.0-dados/dengue_limpo_2025.zip"
# Nome exato do arquivo CSV dentro do .zip
CSV_LIMPO = "dengue_limpo_2025.csv"
# Chama a função para carregar os dados limpos
df_dengue = carregar_dados_dengue_de_release(
    url_release=URL_DADOS_LIMPOS,
    ficheiro_csv_no_zip=CSV_LIMPO
)
df_dengue = df_dengue.dropna()

# Cria df reduzido
df_dengue_reduzido = pd.read_csv('Treinamentos/data/df_dengue_reduzido_20.csv')

# # Roda os experimentos
# if use_gridsearch:
#     # Roda experimentos com GridSearch
#     _, _, _ = treinar_gridsearch(df_dengue_reduzido, target, config)

# if use_optuna:
#     # Roda experimentos com Optuna
#     _, _, _ = treinar_optuna(df_dengue_reduzido, target, config)
