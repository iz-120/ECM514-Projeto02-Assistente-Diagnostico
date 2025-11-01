from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import lightgbm as lgb

def criar_modelo(config):
    """
    Instancia modelo a partir do tipo informado no YAML.
    Todos os modelos são configurados para classificação binária.
    """
    if config['model']['type'].lower() in ['randomforest', 'randomforestclassifier']:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=config['train']['random_state'],
            class_weight='balanced'  # Lida com desbalanceamento
        )
    elif config['model']['type'].lower() in ['xgbclassifier', 'xgboost']:
        return xgb.XGBClassifier(
            random_state=config['train']['random_state'],
            tree_method='hist',
            scale_pos_weight=config['model']['scale_pos_weight'],  # Ajustar baseado no desbalanceamento
            objective='binary:logistic'
        )
    elif config['model']['type'].lower() in ['lgbmclassifier', 'lightgbm']:
        return lgb.LGBMClassifier(
            random_state=config['train']['random_state'],
            objective='binary',
            class_weight='balanced',
        )
    else:
        raise ValueError(f"Modelo {config['model']['type']} não suportado")
    
def aplica_parametros(modelo_base, config):
    if config['model']['param_format'].lower() in ['grid', 'gridsearch', 'gridsearchcv']:        
        return GridSearchCV(
            modelo_base,
            param_grid=config['model']['params'],
            cv=config['train']['cv'],
            scoring=config['grid_search']['scoring'],
            refit=config['grid_search']['refit'],
            n_jobs=config['grid_search']['n_jobs'],
            verbose=config['grid_search']['verbose'],
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