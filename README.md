# Sistema de Apoio à Decisão Clínica

Projeto da disciplina de Ciência de Dados e Inteligência Artificial focado em Machine Learning. O caso de estudo é desenvolver um assistente de diagnóstico de casos graves de dengue.

## Metodologia

### Fonte de dados

O conjunto de dados para este estudo será extraído do Sistema de Informação de Agravos de Notificação (SINAN), a base de dados oficial para o registro de doenças de notificação compulsória no Brasil.

O acesso aos microdados será realizado de forma programática utilizando a biblioteca de código aberto `pysus`, uma ferramenta desenvolvida para facilitar a aquisição e o pré-processamento de dados dos sistemas de informação do Sistema Único de Saúde (SUS). Seriam coletados os dados de notificações de dengue referentes ao período de 2015 a 2024, abrangendo diferentes perfis epidemiológicos sazonais e surtos, a fim de garantir um volume de dados robusto e representativo para o treinamento dos modelos. No entanto, foram coletados e utilizados dados referentes apenas ao ano de 2025 por conta do exorbitante volume de dados (na casa dos milhões de linhas por ano), os quais não estavam sendo suportados por falta de capacidade computacional.

### Pré-processamento dos Dados

**Filtragem e Seleção:** Inicialmente, o dataset será filtrado para incluir apenas registros onde o campo `ID_AGRAVO` corresponda a "Dengue". Em seguida, serão selecionados exclusivamente os casos com diagnóstico confirmado, utilizando-se os códigos pertinentes da variável `CLASSI_FIN`, como "Dengue", "Dengue com Sinais de Alarme" e "Dengue Grave", descartando-se os casos inconclusivos ou não confirmados.

**Definição da Variável Alvo:** A variável alvo do modelo, denominada `RISCO_GRAVIDADE`, será criada a partir da coluna `CLASSI_FIN`. Ela será binária, onde o valor `1` (classe positiva) representará os casos graves (agrupando "Dengue com Sinais de Alarme" e "Dengue Grave") e o valor `0` (classe negativa) representará os casos de "Dengue" clássica, sem sinais de alarme.

**Engenharia de Features:** As variáveis preditoras (_features_) incluirão os sinais e sintomas registrados na notificação que são os mais relevantes no diagnóstico de dengue e na identificação dos casos mais graves (e.g., `FEBRE`, `MIALGIA`, `VOMITO`, `PETEQUIA`), além de dados demográficos como CS_SEXO. A variável `IDADE` será calculada a partir da diferença entre a data dos primeiros sintomas (`DT_SIN_PRI`) e a data de nascimento (`DT_NASC`). Variáveis categóricas serão transformadas em formato numérico através da técnica de _One-Hot Encoding_.

**Tratamento de Dados:** Devido à natureza dos dados de saúde, espera-se a presença de valores ausentes, especialmente nos sintomas, os quais não são de preenchimento obrigatório na Ficha Individual de Notificação (FIN) do SINAN. Em consulta a uma profissional da área, os campos de sintomas vazios indicam a ausência do sintoma em questão e deixa de ser preenchido com “não” por praticidade. Dada a provável desproporção entre casos graves e não graves, o desbalanceamento de classes será mitigado no conjunto de treinamento utilizando a técnica SMOTE (_Synthetic Minority Over-sampling Technique_), que cria amostras sintéticas da classe minoritária para equilibrar a distribuição dos dados.

### Modelagem e Treinamento

Para a tarefa de classificação, serão avaliados e comparados múltiplos algoritmos de _Machine Learning_:

- **Regressão Logística:** Utilizada como um modelo de base (baseline) por sua simplicidade e interpretabilidade.

- **Random Forest:** Um algoritmo de _ensemble_ baseado em árvores de decisão, conhecido por sua robustez e capacidade de lidar com interações complexas entre variáveis.

- **XGBoost (_Extreme Gradient Boosting_):** Um algoritmo de _Gradient Boosting_ altamente otimizado e eficiente, que frequentemente apresenta performance de ponta em competições e aplicações com dados tabulares.

- **MLP:** PREENCHER

Os dados serão divididos em conjuntos de treinamento (80%) e teste (20%). O ajuste de hiperparâmetros dos modelos será realizado utilizando a técnica de validação cruzada (_cross-validation_) no conjunto de treinamento para evitar superajuste (_overfitting_) e garantir a generalização do modelo.

A performance dos modelos será avaliada no conjunto de teste, que permanece intocado durante o treinamento. As métricas de avaliação selecionadas incluem Acurácia, Precisão, Recall, F1-Score e a Área sob a Curva ROC (AUC), com especial atenção ao Recall da classe positiva, pois em um contexto clínico, é mais crítico identificar corretamente os casos verdadeiramente graves, mesmo que isso resulte em mais falsos positivos

### Ferramentas e Tecnologias

- **Desenvolvimento:**

  - Colab (pré-processamento e modelo finais)

  - VSCode (experimentos de otimização dos modelos)

- **Linguagem:** Python

- **Pré-processamento:** Pandas

- **Implementação dos modelos:** Scikit-learn e XGBoost

- **Avaliação das métricas:** Scikit-learn

- **Logging de Experimentos:** Weights & Biases

- **Visualizações:** Matplotlib, Seaborn e Plotly

- **Aplicação:** Streamlit

- **Controle de versão:** GitHub

---

# Como Executar

## Pré Processamento (Colab)

**Observação:** A execução pode demorar vários minutos por conta do acesso ao `pysus`

1. Acessar o [Notebook de Pré-Processamento](https://colab.research.google.com/drive/1DfIQ_N8k0zYLlUbLRIMVvfy-UECGLUkC?usp=sharing)

2. Executar o Notebook completo

## Treinamento dos Modelos Finais (Colab)

**Observação:** A execução pode demorar vários minutos por conta do grande volume de dados

1. Acessa o [Notebook Principal](https://colab.research.google.com/drive/17n8JOVP2fMVOLMQ85lxLj9Uy21ayvE9J?usp=sharing)

2. Executar Notebook completo

## Aplicação

1. Dentro da pasta **App**, no terminal, executar `pip install -r requirements.txt`

2. Dentro da pasta **App**, no terminal, executar `streamlit run app.py`

## Experimentos

O diretório `Treinamentos/` concentra toda a infraestrutura de experimentação sistemática para otimização de hiperparâmetros e comparação de modelos. A arquitetura foi concebida para separar configuração (arquivos YAML), lógica de treinamento (módulos Python) e dados, facilitando a reprodutibilidade e o versionamento de experimentos.

### Estrutura do Diretório

```
Treinamentos/
├── main.py                  # Script principal de execução
├── requirements.txt         # Dependências do ambiente
├── data/                    # Conjuntos de dados reduzidos em formato CSV
├── Experimentos/            # Arquivos de configuração YAML (um por experimento)
└── src/                     # Módulos Python com lógica de treinamento
    ├── train.py             # Funções de treinamento (GridSearch, Optuna, NN)
    ├── models.py            # Criação de modelos e pipelines
    ├── nn_model.py          # Cria e treina o modelo NN_MLP
    └── utils.py             # Funções auxiliares (avaliação, plots, splits)
```

#### Arquivos YAML (Diretório `Experimentos/`)

Cada arquivo YAML parametriza um experimento completo, especificando: tipo de modelo (RandomForest, XGBoost, LightGBM, LogisticRegression, NN_MLP), método de busca de hiperparâmetros (`GridSearchCV`, `RandomizedSearchCV`, `Optuna`), espaço de busca (listas de valores ou distribuições), parâmetros fixos do modelo, configurações de SMOTE (oversampling da classe minoritária), estratégia de validação cruzada, e métricas de avaliação. A seguir, um exemplo de estrutura típica:

```yaml
model:
  type: "LightGBM" # Tipo do modelo
  param_format: "RandomizedSearchCV" # Método de busca
  params: # Espaço de busca (listas de valores)
    n_estimators: [300, 500, 800]
    max_depth: [5, 10, 15, -1]
    learning_rate: [0.01, 0.05, 0.1]
  fixed_params: # Parâmetros fixos
    objective: "binary"
    boosting_type: "gbdt"

smote: # Parâmetros para SMOTE
  sampling_strategy: [0.7, 0.9, 1.0] # Proporção de oversampling (fixo ou lista)
  k_neighbors: [3, 5] # Vizinhos para síntese (fixo ou lista)

train:
  test_size: 0.2 # Proporção de teste
  cv: 3 # Folds de validação cruzada
  random_state: 42 # Seed para reprodutibilidade

cross_val:
  n_iter: 10 # Iterações para RandomizedSearchCV
  n_jobs: 2 # Paralelismo
  scoring: "recall" # Métrica de otimização
  refit: "recall" # Métrica para refit final
  verbose: 1
  n_trials: 5 # Testes para Optuna

file: "lgbm_4" # Identificador do experimento
```

Os parâmetros de modelo e SMOTE declarados como listas são interpretados como espaços de busca: `GridSearchCV`/`RandomizedSearchCV` testam combinações desses valores, enquanto `Optuna` trata cada lista como um conjunto de escolhas categóricas. Valores únicos (não-lista) são aplicados como fixos.

#### Módulos Python (Diretório `src/`)

**`train.py`** – Implementa três funções principais de treinamento:

- **`treinar_gridsearch(df_dengue, target, config, init)`**:
    - Executa busca exaustiva (`GridSearchCV`) ou aleatória (`RandomizedSearchCV`) de hiperparâmetros.
    - Constrói um pipeline imbalanced-learn (`StandardScaler` → `SMOTE` → classificador)
    - Aplica validação cruzada
    - Treina o melhor modelo
    - Avalia no conjunto de teste
    - Rregistra métricas, curvas de aprendizado e importância de features no Weights & Biases.
- **`treinar_optuna(df_dengue, target, config, init)`**:
    - Utiliza otimização bayesiana (`Optuna`) para buscar hiperparâmetros.
    - Lê o espaço de busca do YAML (incluindo parâmetros de SMOTE)
    - Define uma função objetivo que monta o pipeline e executa validação cruzada, maximiza o recall médio, e registra o histórico de trials e os melhores parâmetros no W&B.
- **`treinar_nn(df_dengue, target, config, init)`** (em `nn_model.py`):
    - Implementa treinamento de redes neurais MLP (Multi-Layer Perceptron) com suporte a validação cruzada manual, `GridSearchCV` ou `Optuna`.
    - O pipeline inclui `SimpleImputer` → `MinMaxScaler` → `SMOTE` → `MLPClassifier`, e logs detalhados de recall por fold.

**`models.py`** – Contém a fábrica de modelos (`criar_modelo`) que instancia o classificador base a partir do tipo declarado no YAML (`RandomForest`, `LogisticRegression`, `XGBoost`, `LightGBM`), e a função `aplica_parametros`, que envolve o modelo em um pipeline imbalanced-learn com `StandardScaler` e `SMOTE`, e aplica `GridSearchCV`/`RandomizedSearchCV` quando `param_format` for especificado. Os parâmetros de busca são prefixados com `clf__` (para o classificador) e `smote__` (para o SMOTE), permitindo que o método de busca otimize ambos simultaneamente.

**`utils.py`** – Agrupa funções auxiliares: `define_train_test` divide os dados em treino e teste com estratificação, `avaliar_modelo_completo` computa métricas (acurácia, precisão, recall, F1, AUC), `plot_roc_curve` e `plot_confusion_matrix` geram visualizações registradas no W&B, `plot_learning_curves` calcula curvas de aprendizado, e `flatten_config` achata dicionários aninhados de configuração para melhorar a legibilidade dos logs.

### Passo a Passo para Execução

#### 1. Instalação de Dependências (Primeira Execução)

Navegue até o diretório `Treinamentos/` e instale as bibliotecas necessárias:

```bash
cd Treinamentos
pip install -r requirements.txt
```

#### 2. Configuração do Weights & Biases

Para habilitar o logging de experimentos, crie uma conta gratuita em [wandb.ai](https://wandb.ai) e obtenha sua chave de API. Em seguida, autentique localmente executando:

```bash
wandb login
```

Insira sua chave de API quando solicitado. Alternativamente, descomente e edite a linha no `main.py` (comentar novamente após a primeira execução com login efetuado):

```python
# wandb.login(key='sua_chave_API')
```

#### 3. Configuração do Experimento

Abra o arquivo `main.py` e ajuste as variáveis de configuração.

Vale notar que:
- `org_name`: Se mantido o atual, a run será logada na org de desenvolvimento
- O número em `df_name` representa o % de recorte do dataset original. Foram feitos recortes para otimizar o tempo de execução das runs

**`init`** – Metadados do experimento para o W&B:

```python
init = {
    'org_name': 'seu_usuario_ou_org',          # Nome do usuário ou organização no W&B 
    'project_name': 'Assistente_Diagnostico_Dengue',
    'tags': ['5%'],                            # Tags para filtrar runs
    'name': 'Dengue_v6',                       # Nome único do run
    'df_name': 'df_dengue_reduzido_5'          # Nome do CSV em data/
}
```

**Seleção do arquivo YAML:**

```python
yaml_file = "lgbm_5"  # Nome do arquivo (sem extensão) em Experimentos/
file_path = "Treinamentos/Experimentos/" + yaml_file + ".yaml" # (NÃO alterar)
```

**Definição da variável alvo:**

```python
target = 'RISCO_GRAVIDADE_grave'  # NÃO alterar
```

**Métodos de treinamento a executar:**

```python
use_grid_or_randomized_search = False  # True para GridSearchCV/RandomizedSearchCV
use_optuna = True                      # True para Optuna
use_nn = False                         # True para redes neurais MLP
```

#### 4. Execução do Experimento

Com as configurações definidas, execute o script principal:

```bash
python main.py
```

O script carregará o CSV especificado (`Treinamentos/data/df_dengue_reduzido_5.csv`), lerá o YAML correspondente, e executará os métodos de treinamento habilitados. Todos os resultados (métricas, hiperparâmetros, curvas ROC, matrizes de confusão, importância de features) serão automaticamente registrados no Weights & Biases.

#### 5. Criação de Novos Experimentos

Para testar um novo espaço de hiperparâmetros ou configuração:

1. **Duplique um arquivo YAML existente** em `Experimentos/` e renomeie.
2. **Edite os campos** conforme desejado. Uma boa prática é que o nome em `file`seja equivalente ao nome do arquivo por rastreabilidade nos logs
3. **Atualize `main.py`** para apontar ao novo YAML (`yaml_file = "rf_5"`).
4. **Se desejar**, atualize o `df_name` para ajustar o tamanho da dataset
5. **Se desejar**, ajuste a lista de `tags`
6. **Execute** `python main.py`.

**Dica:** Para testes rápidos, reduza `cross_val.n_iter` (`RandomizedSearchCV`) ou adicione `optuna.n_trials: 5` no YAML (`Optuna`), acelerando a busca e permitindo validar pipelines antes de runs completos.

### Visualização de Resultados

Todos os logs de experimentos podem ser visualizados no painel do Weights & Biases acessando [este link](https://wandb.ai/izabel-sampaio-org/Assistente_Diagnostico_Dengue?nw=nwuserizabelsampaio). O painel permite comparar runs, analisar a evolução de métricas, inspecionar hiperparâmetros ótimos e exportar artefatos (modelos salvos, gráficos, tabelas).
