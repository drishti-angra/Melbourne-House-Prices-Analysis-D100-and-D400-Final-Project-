#%%
#import functions
from re import split
from final_project.preprocessing.preprocessing import load_parquet_data, create_sample_split, save_train_test_parquet, split_X_y, ZeroMedianImputer, UpperWinsorizer, ConditionalMedianImputer, preprocessor
from final_project.modelling_GLM.modelling_GLM import pipeline_glm
from final_project.modelling_LGBM.modelling_LGBM import pipeline_lgbm


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
# Splitting into X_train, y_train
X_train, y_train = split_X_y(df_train, target_col = "log_Price")

# %%
# Splitting into X_test, y_test
X_test, y_test = split_X_y(df_test, target_col = "log_Price")

