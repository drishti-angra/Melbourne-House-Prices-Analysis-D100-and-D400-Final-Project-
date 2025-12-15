from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from final_project.preprocessing.preprocessing import preprocessor, ConditionalMedianImputer




def pipeline_lgbm() -> Pipeline:
    """
    Defines the pipeline for LightGBM model.
    """

    lgbm_model = LGBMRegressor(objective="regression")

    lgbm_pipeline = Pipeline(
        steps=[
        ("conditional_impute", ConditionalMedianImputer(target_col = "YearBuilt", condition_col="Regionname")),
        ("preprocess", preprocessor()),
        ("lgbm_model", lgbm_model)
        ]
    )
    
    return lgbm_pipeline


def tuning_lgbm(
        X_training: pd.DataFrame, 
        y_training: pd.Series, 
        chosen_pipeline: Pipeline,
        cv_folds: int = 5,
        ) -> Pipeline:
    
    param_distributions = {

    }

    cv = RandomizedSearchCV(
        estimator = chosen_pipeline,
        param_distributions= param_distributions,
        n_iter = 20,
        cv = cv_folds,
        scoring = "neg_mean_squared_error", 
        n_jobs = 1, 
        random_state = 42, 
        verbose = 1,
    )

    cv.fit(X_training, y_training)
    best_pipeline_lgbm = cv.best_estimator_

    print("best score:", cv.best_score_)
    print("best pipeline:", best_pipeline_lgbm)
    print("best_parameters:", cv.best_params_)

    return best_pipeline_lgbm