#%%
#import functions
from re import split
from final_project.preprocessing.preprocessing import load_parquet_data, create_sample_split, save_train_test_parquet, split_X_y, ZeroMedianImputer, UpperWinsorizer, ConditionalMedianImputer, preprocessor
from final_project.modelling.modelling_GLM.modelling_GLM import pipeline_glm, tuning_glm
from final_project.modelling.modelling_LGBM.modelling_LGBM import pipeline_lgbm, tuning_lgbm
from final_project.modelling.save_load import save_pipeline, save_X_y


#%% 
# Load cleaned data 

df_compiled = load_parquet_data()
df_compiled.head()

#%%
#Creating a Train Test Split 
create_sample_split(df_compiled, id_column="Address", training_frac=0.8)
df_compiled.head()

# %%
# Save train and test sets to parquet files
df_train, df_test = save_train_test_parquet(df_compiled)
# %%
# Splitting into X_train, y_train and saving it
X_train, y_train = split_X_y(df_train, target_col = "log_Price")
save_X_y(X_train, y_train, "train")

# %%
# Splitting into X_test, y_test and saving it
X_test, y_test = split_X_y(df_test, target_col = "log_Price")
save_X_y(X_test, y_test, "test")

#%%
#Tuning of GLM model (regularisation) to find the best model 
glm_best_pipeline = tuning_glm (X_train, y_train, pipeline_glm())
save_pipeline(glm_best_pipeline, "glm_best_pipeline")

# %%
#Hyperparameter tuning of LightGBM model to find the best model 
lgbm_best_pipeline = tuning_lgbm (X_train, y_train, pipeline_lgbm())
save_pipeline(lgbm_best_pipeline, "lgbm_best_pipeline")

#%%