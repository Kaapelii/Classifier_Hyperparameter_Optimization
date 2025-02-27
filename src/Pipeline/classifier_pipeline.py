from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from Pipeline.feature_union import create_feature_union
from Analytics.heatmap import make_heatmap
import joblib
import config
import time
import os

def create_pipeline():
    feature_union = create_feature_union()
    
    pipeline = Pipeline([
        ('features', feature_union),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', SVC(random_state=config.RANDOM_STATE))
    ])
    
    return pipeline

def create_optimized_pipeline(param_grid, type, X, y):
    pipeline = create_pipeline()
    
    if type == 1:
        grid_search = GridSearchCV(pipeline, 
                                   return_train_score=True, 
                                   param_grid=param_grid, 
                                   cv=config.CROSS_VALIDATIONS, 
                                   n_jobs=config.NJOBS, 
                                   verbose=config.VERBOSE, 
                                   error_score='raise'
                                   )
        search_name = "GridSearchCV"
        is_successive_halving = False
    elif type == 2:
        grid_search = HalvingGridSearchCV(pipeline, 
                                          return_train_score=True, 
                                          param_grid=param_grid, 
                                          cv=config.CROSS_VALIDATIONS, 
                                          n_jobs=config.NJOBS, 
                                          verbose=config.VERBOSE, 
                                          error_score='raise', 
                                          random_state=config.RANDOM_STATE,
                                          )
        search_name = "HalvingGridSearchCV"
        is_successive_halving = True
    elif type == 3:
        grid_search = RandomizedSearchCV(pipeline, 
                                         return_train_score=True, 
                                         param_distributions=param_grid, 
                                         cv=config.CROSS_VALIDATIONS, 
                                         n_jobs=config.NJOBS, 
                                         verbose=config.VERBOSE, 
                                         error_score='raise', 
                                         random_state=config.RANDOM_STATE, 
                                        )
        search_name = "RandomizedSearchCV"
        is_successive_halving = False
    elif type == 4:
        grid_search = HalvingRandomSearchCV(pipeline, 
                                            return_train_score=True, 
                                            param_distributions=param_grid, 
                                            cv=config.CROSS_VALIDATIONS, 
                                            n_jobs=config.NJOBS, 
                                            verbose=config.VERBOSE, 
                                            error_score='raise', 
                                            random_state=config.RANDOM_STATE
                                            )
        search_name = "HalvingRandomSearchCV"
        is_successive_halving = True
    else:
        return None, "No valid type provided", 0
    
    start_time = time.time()
    grid_search.fit(X, y)  
    end_time = time.time()
    time_spent = end_time - start_time
    print(f"Time spent on {search_name}: {time_spent}")
    joblib.dump(grid_search.best_estimator_, os.path.join(config.PIPELINE_DIR, 'pipeline_' + search_name + '.pkl'))

    if config.CREATE_HEATMAP:
        make_heatmap(grid_search, time_spent, is_successive_halving, search_name)
    
    return grid_search, search_name, time_spent