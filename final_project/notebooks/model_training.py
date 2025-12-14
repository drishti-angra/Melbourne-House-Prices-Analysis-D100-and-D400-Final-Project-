#%%
#import functions
from final_project.preprocessing.preprocessing import load_parquet_data, create_sample_split, save_train_test_parquet



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

