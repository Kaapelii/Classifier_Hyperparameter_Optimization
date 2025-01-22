from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from Pipeline.feature_union import create_feature_union
import joblib
import config

def create_pipeline():
    feature_union = create_feature_union()
    
    pipeline = Pipeline([
        ('features', feature_union),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', SVC(random_state=config.RANDOM_STATE))
    ])
    
    return pipeline

def create_optimized_pipeline(param_grid, type):
    pipeline = create_pipeline()
    
    if type == 1 or type == 5:
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise', random_state=config.RANDOM_STATE)
        return grid_search, "GridSearchCV"
    
    if type == 2 or type == 5:
        grid_search = HalvingGridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise', random_state=config.RANDOM_STATE)
        return grid_search, "HalvingGridSearchCV"
    
    if type == 3 or type == 5:
        grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise', random_state=config.RANDOM_STATE)
        return grid_search, "RandomizedSearchCV"
    
    if type == 4 or type == 5:
        grid_search = HalvingRandomSearchCV(pipeline, param_distributions=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise', random_state=config.RANDOM_STATE)
        return grid_search, "HalvingRandomSearchCV"
    
    return grid_search

