from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from lightgbm import LGBMRegressor
from final_project.preprocessing.preprocessing import preprocessor, ConditionalMedianImputer

import lightgbm as lgb
import pandas as pd


def pipeline_lgbm() -> Pipeline:
    """
    Defines the pipeline for LightGBM model.

    """

    lgbm_model = LGBMRegressor(objective="regression", n_jobs=1)

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
        validation_size: float = 0.2,
        early_stopping_rounds: int = 25,
        random_state: int = 42,
        ) -> Pipeline:
    
    """
    Hyperparameter tuning for the LightGBM model.

    Parameters
    ----------
    X_training : pd.DataFrame
        Training features.
    y_training : pd.Series
        Training target.
    chosen_pipeline : Pipeline
        A scikit-learn Pipeline.
    cv_folds : int, default=5
        Number of cross-validation folds.
    validation_size: float, default = 0.2
        Proportion of training data used for the early stopping validations set.
    early_stopping_rounds: int, default = 25
        number of rounds without improvement before applying early stopping.
    random_state: int, default = 42
        For reproducibility of results.

    Returns
    -------
    Pipeline
        Best LGBM pipeline after tuning
    

    """


    X_tr, X_val, y_tr, y_val = train_test_split(
        X_training,
        y_training, 
        test_size= validation_size,
        random_state= random_state
    )
    

    param_distributions = {
        "lgbm_model__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
        "lgbm_model__n_estimators": [5000],
        "lgbm_model__num_leaves": [15, 30, 60, 120],
        "lgbm_model__min_child_weight": [0.1, 1, 2, 5],
    }

    cv = RandomizedSearchCV(
        estimator = chosen_pipeline,
        param_distributions= param_distributions,
        n_iter = 20,
        cv = cv_folds,
        scoring = "neg_mean_squared_error", 
        n_jobs = 1, 
        random_state = random_state, 
        verbose = 1,
    )


    cv.fit(
        X_tr,
        y_tr,
        lgbm_model__eval_set=[(X_val, y_val)],
        lgbm_model__callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
    )



    best_pipeline_lgbm = cv.best_estimator_

    print("best score:", cv.best_score_)
    print("best pipeline:", best_pipeline_lgbm)
    print("best_parameters:", cv.best_params_)

    return best_pipeline_lgbm