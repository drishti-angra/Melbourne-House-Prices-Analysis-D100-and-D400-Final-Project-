from sklearn.pipeline import Pipeline
from glum import GeneralizedLinearRegressor, TweedieDistribution
from final_project.preprocessing.preprocessing import preprocessor, ConditionalMedianImputer


def pipeline_glm() -> Pipeline:
    """
    Defines the pipeline for GLM model.
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