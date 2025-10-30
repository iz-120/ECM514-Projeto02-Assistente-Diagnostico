from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import lightgbm as lgb

def criar_modelo(model_type, random_state=None):
    """
    Instancia modelo a partir do tipo informado no YAML.
    Todos os modelos são configurados para classificação binária.
    """
    if model_type.lower() in ['randomforest', 'randomforestclassifier']:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced'  # Lida com desbalanceamento
        )
    elif model_type.lower() in ['xgbclassifier', 'xgboost']:
        return xgb.XGBClassifier(
            random_state=random_state,
            tree_method='hist',
            scale_pos_weight=1,  # Ajustar baseado no desbalanceamento
            objective='binary:logistic'
        )
    elif model_type.lower() in ['lgbmclassifier', 'lightgbm']:
        return lgb.LGBMClassifier(
            random_state=random_state,
            objective='binary',
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Modelo {model_type} não suportado")
    
def aplica_parametros(modelo_base, config):
    if config['model']['param_format'].lower() in ['grid', 'gridsearch', 'gridsearchcv']:
        # Define múltiplos scorings para classificação
        scoring = {
            'roc_auc': 'roc_auc',           # Área sob a curva ROC
            'f1': 'f1',                      # Equilíbrio entre precisão e recall
            'bal_acc': 'balanced_accuracy'   # Média entre sensibilidade e especificidade
        }
        
        return GridSearchCV(
            modelo_base,
            param_grid=config['model']['params'],
            cv=config['train']['cv'],
            scoring=scoring,
            refit='roc_auc',  # Usa ROC AUC para selecionar melhor modelo
            n_jobs=2,
            verbose=1,
            return_train_score=True
        )
    elif config['model']['param_format'].lower() in ['simple', 'finetuning']:
        return modelo_base(
            n_estimators=config['model']['params']['n_estimators'],
            max_depth=config['model']['params']['max_depth'],
            min_samples_split=config['model']['params']['min_samples_split'],
            min_samples_leaf=config['model']['params']['min_samples_leaf'],
            max_features=config['model']['params']['max_features']
        )