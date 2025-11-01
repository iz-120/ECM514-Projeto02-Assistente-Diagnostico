from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def criar_modelo(config):
    """
    Instancia modelo a partir do tipo informado no YAML.
    Todos os modelos são configurados para classificação binária.
    """
    if config['model']['type'].lower() in ['randomforest', 'randomforestclassifier']:
        return RandomForestClassifier(
            random_state=config['train']['random_state'],
            class_weight='balanced'  # Lida com desbalanceamento
        )
    elif config['model']['type'].lower() in ['logistic', 'logisticregression', 'logistic_regression']:
        return LogisticRegression(
            random_state=config['train']['random_state'],
            solver=config['model']['params']['solver'],
            max_iter=config['model']['params']['max_iter'],
            penalty=config['model']['params']['penalty'],
            C=config['model']['params']['C']
        )
    elif config['model']['type'].lower() in ['xgbclassifier', 'xgboost']:
        return xgb.XGBClassifier(
            random_state=config['train']['random_state'],
            tree_method=config['model']['fixed_params']['tree_method'],
            objective=config['model']['fixed_params']['objective']
        )
    elif config['model']['type'].lower() in ['lgbmclassifier', 'lightgbm']:
        return lgb.LGBMClassifier(
            random_state=config['train']['random_state'],
            objective=config['model']['fixed_params']['objective'],
            class_weight=config['model']['fixed_params']['class_weight'],
        )
    else:
        raise ValueError(f"Modelo {config['model']['type']} não suportado")
    
def aplica_parametros(modelo_base, config):
    # Build a pipeline that applies scaling and SMOTE before the classifier
    pipeline = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=config['train'].get('random_state', 42))),
        ('clf', modelo_base)
    ])

    if config['model']['param_format'].lower() in ['grid', 'gridsearch', 'gridsearchcv']:
        # transform param grid keys to target the classifier inside the pipeline
        raw_grid = config['model'].get('params', {})
        param_grid = {}
        for k, v in raw_grid.items():
            param_grid[f'clf__{k}'] = v

        return GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=config['train']['cv'],
            scoring=config['cross_val']['scoring'],
            refit=config['cross_val'].get('refit', True),
            n_jobs=config['cross_val'].get('n_jobs', 1),
            verbose=config['cross_val'].get('verbose', 0),
            return_train_score=True
        )
    elif config['model']['param_format'].lower() in ['simple', 'finetuning']:
        # apply provided params to the classifier and return the full pipeline
        params = config['model'].get('params', {})
        try:
            modelo_base.set_params(**params)
        except Exception:
            # some configs may already be full objects; ignore failures and use base
            pass
        pipeline.set_params(clf=modelo_base)
        return pipeline