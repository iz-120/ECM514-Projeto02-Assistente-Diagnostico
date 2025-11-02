# Sistema de Apoio à Decisão Clínica
Projeto da disciplina de Ciência de Dados e Inteligência Artificial focado em Machine Learning. O caso de estudo é desenvolver um assistente de diagnóstico de casos graves de dengue.

## Resumo 

INSERIR RESUMO

## Metodologia
### Fonte de dados

O conjunto de dados para este estudo será extraído do Sistema de Informação de Agravos de Notificação (SINAN), a base de dados oficial para o registro de doenças de notificação compulsória no Brasil.

O acesso aos microdados será realizado de forma programática utilizando a biblioteca de código aberto `pysus`, uma ferramenta desenvolvida para facilitar a aquisição e o pré-processamento de dados dos sistemas de informação do Sistema Único de Saúde (SUS). Seriam coletados os dados de notificações de dengue referentes ao período de 2015 a 2024, abrangendo diferentes perfis epidemiológicos sazonais e surtos, a fim de garantir um volume de dados robusto e representativo para o treinamento dos modelos. No entanto, foram coletados e utilizados dados referentes apenas ao ano de 2025 por conta do exorbitante volume de dados (na casa dos milhões de linhas por ano), os quais não estavam sendo suportados por falta de capacidade computacional.

### Pré-processamento dos Dados

**Filtragem e Seleção:** Inicialmente, o dataset será filtrado para incluir apenas registros onde o campo `ID_AGRAVO` corresponda a "Dengue". Em seguida, serão selecionados exclusivamente os casos com diagnóstico confirmado, utilizando-se os códigos pertinentes da variável `CLASSI_FIN`, como "Dengue", "Dengue com Sinais de Alarme" e "Dengue Grave", descartando-se os casos inconclusivos ou não confirmados.

**Definição da Variável Alvo:** A variável alvo do modelo, denominada `RISCO_GRAVIDADE`, será criada a partir da coluna `CLASSI_FIN`. Ela será binária, onde o valor `1` (classe positiva) representará os casos graves (agrupando "Dengue com Sinais de Alarme" e "Dengue Grave") e o valor `0` (classe negativa) representará os casos de "Dengue" clássica, sem sinais de alarme.

**Engenharia de Features:** As variáveis preditoras (*features*) incluirão os sinais e sintomas registrados na notificação que são os mais relevantes no diagnóstico de dengue e na identificação dos casos mais graves (e.g., `FEBRE`, `MIALGIA`, `VOMITO`, `PETEQUIA`), além de dados demográficos como CS_SEXO. A variável `IDADE` será calculada a partir da diferença entre a data dos primeiros sintomas (`DT_SIN_PRI`) e a data de nascimento (`DT_NASC`). Variáveis categóricas serão transformadas em formato numérico através da técnica de *One-Hot Encoding*.

**Tratamento de Dados:** Devido à natureza dos dados de saúde, espera-se a presença de valores ausentes, especialmente nos sintomas, os quais não são de preenchimento obrigatório na Ficha Individual de Notificação (FIN) do SINAN. Em consulta a uma profissional da área, os campos de sintomas vazios indicam a ausência do sintoma em questão e deixa de ser preenchido com “não” por praticidade. Dada a provável desproporção entre casos graves e não graves, o desbalanceamento de classes será mitigado no conjunto de treinamento utilizando a técnica SMOTE (*Synthetic Minority Over-sampling Technique*), que cria amostras sintéticas da classe minoritária para equilibrar a distribuição dos dados.


### Modelagem e Treinamento

Para a tarefa de classificação, serão avaliados e comparados múltiplos algoritmos de *Machine Learning*:

* **Regressão Logística:** Utilizada como um modelo de base (baseline) por sua simplicidade e interpretabilidade.

* **Random Forest:** Um algoritmo de *ensemble* baseado em árvores de decisão, conhecido por sua robustez e capacidade de lidar com interações complexas entre variáveis.

* **XGBoost (*Extreme Gradient Boosting*):** Um algoritmo de *Gradient Boosting* altamente otimizado e eficiente, que frequentemente apresenta performance de ponta em competições e aplicações com dados tabulares.

* **MLP:** PREENCHER

Os dados serão divididos em conjuntos de treinamento (80%) e teste (20%). O ajuste de hiperparâmetros dos modelos será realizado utilizando a técnica de validação cruzada (*cross-validation*) no conjunto de treinamento para evitar superajuste (*overfitting*) e garantir a generalização do modelo.

A performance dos modelos será avaliada no conjunto de teste, que permanece intocado durante o treinamento. As métricas de avaliação selecionadas incluem Acurácia, Precisão, Recall, F1-Score e a Área sob a Curva ROC (AUC), com especial atenção ao Recall da classe positiva, pois em um contexto clínico, é mais crítico identificar corretamente os casos verdadeiramente graves, mesmo que isso resulte em mais falsos positivos


### Ferramentas e Tecnologias

* **Desenvolvimento:** ADICIONAR NO ARTIGO

    * Colab (pré-processamento e modelo finais)

    * VSCode (experimentos de otimização dos modelos)

* **Linguagem:** Python

* **Pré-processamento:** Pandas

* **Implementação dos modelos:** Scikit-learn e XGBoost

* **Avaliação das métricas:** Scikit-learn

* **Logging de Experimentos:** Weights & Biases ADICIONAR NO ARTIGO

* **Visualizações:** Matplotlib e Seaborn

* **Aplicação:** Streamlit

* **Controle de versão:** GitHub

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

NO ARTIGO, EXPLICAR ESTRUTURA

1. Dentro da pasta **App**, no terminal, executar `pip install -r requirements.txt`

2. Dentro da pasta **App**, no terminal, executar `streamlit run app.py`

## Experimentos

NO ARTIGO, EXPLICAR ESTRUTURA DOS TREINAMENTOS

CRIAR REQUIREMENTS.TXT PARA PERMITIR QUE QUALQUER UM EXECUTE O PROJETO

TROCAR LOGIN DO WANDB

EXPLICAR INPUTS

Caso queira visualizar os logs de todos os experimentos executados, basta acessar o projeto na plataforma do Weights & Biases com [este link](https://wandb.ai/izabel-sampaio-org/Assistente_Diagnostico_Dengue?nw=nwuserizabelsampaio)

---

# Referências
