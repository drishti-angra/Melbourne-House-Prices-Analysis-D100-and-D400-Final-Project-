from sklearn.pipeline import Pipeline
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
