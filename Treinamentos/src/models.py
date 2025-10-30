from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import lightgbm as lgb

def criar_modelo(model_type, random_state=None):
    """Instancia modelo a partir do tipo informado no YAML"""
    if model_type.lower() in ['randomforest', 'randomforestregressor']:
        return RandomForestRegressor(random_state=random_state)
    elif model_type.lower() in ['xgbregressor', 'xgboost']:
        return xgb.XGBRegressor(random_state=random_state,
                                tree_method='hist')
    elif model_type.lower() in ['lgbmregressor', 'lightgbm']:
        return lgb.LGBMRegressor(random_state=random_state)
    else:
        raise ValueError(f"Modelo {model_type} não suportado")
    
def aplica_parametros(modelo_base, config):
    if config['model']['param_format'].lower() in ['grid', 'gridsearch', 'gridsearchcv']:
        return GridSearchCV(modelo_base, param_grid=config['model']['params'],
                          cv=config['train']['cv'], scoring='neg_mean_absolute_error',
                          n_jobs=2)
    elif config['model']['param_format'].lower() in ['simple', 'finetuning']:
        return modelo_base(
            n_estimators=config['model']['params']['n_estimators'],
            max_depth=config['model']['params']['max_depth'],
            min_samples_split=config['model']['params']['min_samples_split'],
            min_samples_leaf=config['model']['params']['min_samples_leaf'],
            max_features=config['model']['params']['max_features']
        )