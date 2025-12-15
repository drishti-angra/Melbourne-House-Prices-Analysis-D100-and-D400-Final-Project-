from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from final_project.notebooks.model_training import X_train
from glum import GeneralizedLinearRegressor, TweedieDistribution
from final_project.preprocessing.preprocessing import preprocessor, ConditionalMedianImputer

import pandas as pd


def pipeline_glm() -> Pipeline:
    """
    Defines the pipeline for GLM model. 
    The Gaussian GLM with identity link on log Prices is chosen.
    This is MSE loss function on log Prices
    """

    glm_model = GeneralizedLinearRegressor(
        family = TweedieDistribution(power=0.0), 
        fit_intercept=True
    )

    glm_pipeline = Pipeline(
        steps=[
        ("conditional_impute", ConditionalMedianImputer(target_col = "YearBuilt", condition_col="Regionname")),
        ("preprocess", preprocessor()),
        ("glm_model", glm_model)
        ]
    )
    
    return glm_pipeline

def tuning_glm(
        X_training: pd.DataFrame, 
        y_training: pd.Series, 
        chosen_pipeline: Pipeline,
        cv_folds: int = 5
        ) -> Pipeline:
    
    """
    Tuning to find the right degree of regularisation for the GLM model.

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
    n_iter : int, default=20
        Number of random hyperparameter combinations to evaluate.

    Returns
    -------
    Pipeline
        Best GLM pipeline after tuning

    """


    param_distributions = {
        "glm_model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "glm_model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    cv = RandomizedSearchCV(
        estimator = chosen_pipeline,
        param_distributions = param_distributions,
        n_iter = 20,
        cv = cv_folds, 
        scoring = "neg_mean_squared_error", 
        n_jobs = 1, 
        random_state = 42, 
        verbose = 1,
    )

    cv.fit(X_training, y_training)
    best_pipeline = cv.best_estimator_

    print("best score:", cv.best_score_)
    print("best pipeline:", best_pipeline)
    print("best_parameters:", cv.best_params_)

    return best_pipeline