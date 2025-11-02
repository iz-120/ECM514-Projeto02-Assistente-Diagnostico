'''
app.py

Dashboard interativo com Streamlit para predição de risco de Dengue Grave.

Para executar, abra o terminal e corra:
streamlit run app.py
'''

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =============================================================================
# CARREGAMENTO DOS ARTEFACTOS DO MODELO
# =============================================================================

# Usar o cache do Streamlit para carregar o modelo e o explicador apenas uma vez
@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de inferência (Scaler + Modelo)."""
    try:
        pipeline = joblib.load('melhor_modelo_dengue.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("ERRO: Ficheiro 'melhor_modelo_dengue.pkl' não encontrado.")
        return None

@st.cache_resource
def carregar_explicador():
    """Carrega o explicador SHAP."""
    try:
        explainer = joblib.load('shap_explainer.pkl')
        return explainer
    except FileNotFoundError:
        st.error("ERRO: Ficheiro 'shap_explainer.pkl' não encontrado.")
        return None

@st.cache_resource
def carregar_colunas():
    """Carrega a lista de colunas."""
    try:
        colunas = joblib.load('colunas_modelo.pkl')
        return colunas
    except FileNotFoundError:
        st.error("ERRO: Ficheiro 'colunas_modelo.pkl' não encontrado.")
        return None

# Carregar os artefactos
model_pipeline = carregar_modelo()
shap_explainer = carregar_explicador()
model_colunas = carregar_colunas()

# =============================================================================
# INTERFACE DO UTILIZADOR (SIDEBAR PARA INPUTS)
# =============================================================================

st.set_page_config(layout="wide")
st.title("DEV-Dengue: Sistema de Apoio à Decisão Clínica")
st.markdown("Insira os dados do paciente na barra lateral para obter uma predição de risco.")

# A barra lateral é ideal para colocar os controlos de input
st.sidebar.header("Informações do Paciente")

# --- Widgets de Input ---
# Adapte os valores min/max e as opções conforme o seu dataset

# Inputs Demográficos
idade = st.sidebar.number_input("Idade", min_value=0, max_value=120, value=30, step=1)
dias_com_sintomas = st.sidebar.number_input("Dias com sintomas", min_value=0, max_value=120, value=1, step=1)
sexo = st.sidebar.selectbox("Sexo", options=[('Masculino', 0), ('Feminino', 1), ('Não declarar', 2)], format_func=lambda x: x[0])

# Inputs de Sinais e Sintomas (Checkboxes são perfeitos para isto)
st.sidebar.subheader("Sinais e Sintomas Clínicos")

# Crie um checkbox para cada sintoma que o seu modelo usa
# (os nomes devem corresponder ao que está em 'colunas_modelo.pkl')
febre = st.sidebar.checkbox("Febre")
mialgia = st.sidebar.checkbox("Mialgia (Dor Muscular)")
exantema = st.sidebar.checkbox("Exantema (Erupção Cutânea)")
cefaleia = st.sidebar.checkbox("Cefaleia (Dor de Cabeça)")
vomito = st.sidebar.checkbox("Vômito")
nausea = st.sidebar.checkbox("Náusea")
petequia = st.sidebar.checkbox("Petéquias")
atralgia = st.sidebar.checkbox("Artralgia (Dor nas Articulações)")
dor_retro = st.sidebar.checkbox("Dor Retroorbital")

# Botão para executar a predição
botao_predicao = st.sidebar.button("Avaliar Risco")

# =============================================================================
# LÓGICA DE PREDIÇÃO E EXIBIÇÃO
# =============================================================================

# Esta parte só é executada quando o botão é pressionado
if botao_predicao and model_pipeline is not None and shap_explainer is not None and model_colunas is not None:
    
    # 1. Criar o DataFrame de Input
    # Montar um dicionário com os dados do paciente
    dados_paciente = {
        'IDADE_ANOS': [idade],
        'DIAS_COM_SINTOMAS': [dias_com_sintomas],
        'CS_SEXO_F': [1 if sexo[1] == 1 else 0],
        'CS_SEXO_I': [1 if sexo[1] == 2 else 0],
        'FEBRE_sim': [int(febre)],
        'MIALGIA_sim': [int(mialgia)],
        'CEFALEIA_sim': [int(cefaleia)],
        'EXANTEMA_sim': [int(exantema)],
        'VOMITO_sim': [int(vomito)],
        'NAUSEA_sim': [int(nausea)],
        'PETEQUIA_N_sim': [int(petequia)],
        'ARTRALGIA_sim': [int(atralgia)],
        'DOR_RETRO_sim': [int(dor_retro)]
    }
    
    # Criar o DataFrame com as colunas na ordem correta
    try:
        input_df = pd.DataFrame(dados_paciente)
        # Garantir que todas as colunas do modelo estão presentes, na ordem correta
        input_df = input_df.reindex(columns=model_colunas, fill_value=0) 
    except Exception as e:
        st.error(f"Erro ao formatar os dados de entrada: {e}")
        st.stop() # Para a execução

    # 2. Fazer a Predição
    try:
        predicao = model_pipeline.predict(input_df)[0]
        probabilidade = model_pipeline.predict_proba(input_df)[0][1] # Probabilidade de ser classe 1 (Grave)
    except Exception as e:
        st.error(f"Erro ao executar a predição: {e}")
        st.stop()

    # 3. Exibir os Resultados Principais
    st.subheader("Resultado da Avaliação de Risco")
    
    col1, col2 = st.columns(2)
    
    if predicao == 1:
        col1.error("**ALERTA: Alto Risco de Gravidade**")
        col2.metric(label="Score de Risco (Prob. de Agravamento)", value=f"{probabilidade * 100:.1f}%")
        col1.markdown("Recomenda-se observação clínica intensiva e hidratação venosa, conforme protocolos (Grupo C/D).")
    else:
        col1.success("**Resultado: Baixo Risco de Gravidade**")
        col2.metric(label="Score de Risco (Prob. de Agravamento)", value=f"{probabilidade * 100:.1f}%")
        col1.markdown("Recomenda-se tratamento ambulatorial com hidratação oral e acompanhamento (Grupo A/B).")
    
    st.markdown("---")
    
    # 4. Exibir a Explicação (SHAP)
    st.subheader("Fatores que Contribuíram para a Predição")
    st.markdown("""
    O gráfico abaixo (Waterfall Plot) mostra como cada sintoma e dado do paciente contribuiu
    para "empurrar" o score de risco final, partindo de um valor base (a média de predição do modelo).
    - **Fatores em <span style='color:red;'>vermelho</span>** aumentaram o risco.
    - **Fatores em <span style='color:blue;'>azul</span>** diminuíram o risco.
    """, unsafe_allow_html=True)
    
    try:
        # O SHAP precisa dos dados DEPOIS do Scaler
        # 1. Obter o scaler do pipeline
        scaler = model_pipeline.named_steps['scaler']
        # 2. Transformar os dados de input
        input_scaled = scaler.transform(input_df)
        
        # 3. Calcular os valores SHAP para esta predição específica
        # (O resultado é uma lista, pegamos o primeiro elemento [0])
        shap_values = shap_explainer.shap_values(input_scaled)[0]
        
        # 4. Criar o gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        # Criar o objeto de explicação SHAP
        expl = shap.Explanation(
            values=shap_values,
            base_values=shap_explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=model_colunas
        )
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro ao gerar o gráfico de explicação (SHAP): {e}")

else:
    st.info("A aguardar a inserção dos dados do paciente na barra lateral...")