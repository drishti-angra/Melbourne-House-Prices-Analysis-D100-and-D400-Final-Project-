# house_prices_final_project
Fundamentals of Data Science and Research Computing Combined Project
BGN 1955C


## Installation Instructions 
To install the package, first ensure you are in the parent directory (project root): `d100_d400_code_1955C`

### Then create and activate the virtual environment: 

```
conda env create -f environment.yaml 
```
```
conda activate d100_housing
```

### Then, install the package: 

```
pip install -e . 
```

### Precommit hooks installation: 

```
pre-commit install 
```


## Running the package instructions
The package can be run from the terminal or directly from executing notebooks and python scripts.

### eda_cleaning.ipynb 
This notebook loads the raw data, conducts exploratory data analysis and contains explanatory data analysis plots which can be found in the report. 
It also saves the cleaned data (before feature engineering) as a parquet file under the folder: `d100_d400_code_1955C/data`
To run this notebook from the terminal: 

```
jupyter nbconvert --to notebook --execute notebooks_and_scripts/eda_cleaning.ipynb --inplace
```

### model_training.py
This script loads the parquet file containing cleaned data (before feature engineering). It creates a Train-Test split, saving the training and testing datasets as parquet files. X_train, y_train, X_test and y_test are also saved as parquet files. These are all saved in `d100_d400_code_1955C/data`.

Then, the GLM and LGBM pipelines are tuned to find the best pipeline. The best pipelines are saved under `d100_d400_code_1955C/pipelines`. 

To run this script from the terminal: 
```
python notebooks_and_scripts/model_training.py
```

#### My Own Scikit-learn Transformer

I have created my own scikit-learn transformer in `d100_d400_code_1955C/final_project/preprocessing/feature_engineering.py`. The pytest for this can be found in `d100_d400_code_1955C/tests/test_custombinarizer.py` To run the test from the terminal: 
```
pytest tests/test_custombinarizer.py
```

### evaluation.ipynb
This notebook evalutes the predictions of the tuned GLM and LGBM models. It loafs the best pipelines as well as the X_train, y_train, X_test and y_test parquet files. It generates a table of predictions and creates plots which can be found in the report. The plots are: predicted vs actual plots, feature importance plots and PDP plots using Dalex Explainer class. It also contains a lorenz curve. 

To run this notebook from the terminal: 
```
jupyter nbconvert --to notebook --execute notebooks_and_scripts/evaluation.ipynb --inplace
```
