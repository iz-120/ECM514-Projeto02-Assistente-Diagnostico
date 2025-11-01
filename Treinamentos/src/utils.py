import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# PLOTA ROC CURVE
# ==============================================================================
def plot_roc_curve(y_true, y_pred_proba):
    """
    Plota curva ROC, calcula AUC e loga dados no wandb
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot principal
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    
    # Dados da curva para wandb
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    wandb.log({
        "dados/roc_curve_points": wandb.Table(dataframe=roc_data)
    })
    
    return roc_auc

# ==============================================================================
# PLOTA MATRIZ DE CONFUSÃO
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, labels=['Não Grave', 'Grave']):
    """
    Plota matriz de confusão com números e percentuais
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    return cm

# ==============================================================================
# CALCULA MÉTRICAS BÁSICAS
# ==============================================================================
def avaliar_classificacao_basica(y_true, y_pred):
    """
    Métricas básicas de classificação binária
    """
    metricas = {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred),  # Para classe positiva (grave)
        'Recall/Sensibilidade': recall_score(y_true, y_pred),  # Taxa de detecção de casos graves
        'F1-Score': f1_score(y_true, y_pred)  # Média harmônica entre precisão e recall
    }
    return metricas

# ==============================================================================
# CALCULA MÉTRICAS MÉDICAS
# ==============================================================================
def metricas_medicas(y_true, y_pred, y_pred_proba=None):
    """
    Métricas relevantes para diagnóstico médico
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metricas = {
        'Sensibilidade': tp / (tp + fn),  # Capacidade de detectar casos graves
        'Especificidade': tn / (tn + fp),  # Capacidade de identificar casos não graves
        'VPP': tp / (tp + fp),  # Valor Preditivo Positivo
        'VPN': tn / (tn + fn),  # Valor Preditivo Negativo
        'Taxa_Falsos_Positivos': fp / (fp + tn),  # Qtos não graves foram classificados como graves
        'Taxa_Falsos_Negativos': fn / (fn + tp)   # Qtos graves não foram detectados
    }
    
    # Adicionar AUC se probabilidades disponíveis
    if y_pred_proba is not None:
        metricas['AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metricas

# ==============================================================================
# CALCULA MÉTRICAS COMPLETAS
# ==============================================================================
def avaliar_modelo_completo(y_true, y_pred, y_pred_proba=None, nome_modelo="Modelo"):
    """
    Avaliação completa do modelo de classificação com logging no wandb
    """
    # Métricas básicas
    print(f"\n=== Avaliação do {nome_modelo} ===")
    print("\nMétricas Básicas:")
    metricas_basicas = avaliar_classificacao_basica(y_true, y_pred)
    for metrica, valor in metricas_basicas.items():
        print(f"{metrica}: {valor:.4f}")
        wandb.log({f"metricas/classificacao/{metrica}": valor})
    
    # Matriz de Confusão
    print("\nMatriz de Confusão:")
    plt.figure(figsize=(8, 6))
    cm = plot_confusion_matrix(y_true, y_pred)
    wandb.log({"visualizacoes/matriz_confusao": wandb.Image(plt)})
    plt.close()
    
    # Métricas médicas detalhadas
    print("\nMétricas para Diagnóstico:")
    metricas_med = metricas_medicas(y_true, y_pred, y_pred_proba)
    for metrica, valor in metricas_med.items():
        print(f"{metrica}: {valor:.4f}")
        wandb.log({f"metricas/diagnostico/{metrica}": valor})
    
    # Curva ROC e Precision-Recall se probabilidades disponíveis
    if y_pred_proba is not None:
        # ROC Curve
        plt.figure(figsize=(8, 6))
        roc_auc = plot_roc_curve(y_true, y_pred_proba)
        wandb.log({
            "metricas/roc/AUC": roc_auc,
            "visualizacoes/roc_curve": wandb.Image(plt)
        })
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.legend(loc="lower left")
        wandb.log({
            "metricas/pr/average_precision": avg_precision,
            "visualizacoes/pr_curve": wandb.Image(plt)
        })
        plt.close()
    
    # Resumo de métricas em uma única tabela
    metricas_table = pd.DataFrame({
        'Metrica': list(metricas_basicas.keys()) + list(metricas_med.keys()),
        'Valor': list(metricas_basicas.values()) + list(metricas_med.values())
    })
    wandb.log({
        "tabelas/metricas_resumo": wandb.Table(dataframe=metricas_table)
    })
    
    return {**metricas_basicas, **metricas_med}

# ==============================================================================
# PLOTA GRÁFICOS DE REAL VC PREVISTO
# ==============================================================================
def plotar_real_vs_previsto(datas, y_real, y_previsto, produto_nome, tipo_modelo, tuning, file):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datas, y=y_real, mode='lines+markers', name='Real', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=datas, y=y_previsto, mode='lines+markers', name='Previsto', line=dict(color='red', dash='dot')))
    titulo = f"Real x Previsto - {produto_nome} | Modelo: {tipo_modelo} - {file} - {tuning}"
    fig.update_layout(title=titulo, xaxis_title='Data', yaxis_title='Quantidade vendida', template='plotly_white')
    wandb.log({f"Real_vs_Previsto_{produto_nome}": fig})

# ==============================================================================
# DEFINE TREINO E TESTE
# ==============================================================================
def define_train_test(df_dengue, target, config):
    """
    Separa treino e teste do dataset, usando datas externas (pois df não tem coluna 'DATA').
    
    Parâmetros
    ----------
    df_produto : pd.DataFrame
        Dataset do produto sem a coluna de datas.
    percent_split : dict, opcional
        Configuração contendo 'train' -> 'test_size' para fallback percentual.

    Retorna
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    # Definir X (features) e y (alvo)
    X = df_dengue.drop(target, axis=1)
    y = df_dengue[target]

    # SPLIT PERCENTUAL
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['train']['test_size'],
    random_state=config['train']['random_state'],
    stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==============================================================================
# REMOVE ANINHAMENTO DOS PARÂMETROS EM CONFIG NO WANDB
# ==============================================================================
def flatten_config(cfg, parent_key='', sep='_'):
    """
    Flatten um dicionário aninhado, concatenando as chaves com o separador especificado.
    Ex: {'model': {'params': {'learning_rate': 0.1}}} -> {'model_params_learning_rate': 0.1}
    """
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)