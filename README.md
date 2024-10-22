# IsBadBy
## Work on Keggle competition Don't Get Kicked

- title: Don't Get Kicked!
- authors: faysal, Will Adams and Will Cukierski
- url: \url{https://kaggle.com/competitions/DontGetKicked}

### Instalation
- clone reprository 
- download data from https://www.kaggle.com/c/DontGetKicked/data?select=test.csv
- unpack files to  to /data/
- install enviroment from environment.yaml
- copy notebooks from /notebooks_clean/  to project folder or add coresponded cde lines from path_init.ipynb to the notebooks head
- run notebooks in the marked order (or make train-test split with file names pointed in /data/readme_data.txt)


### Results

In current vertion the Base Model provides best results, notebook: 04_BaseModel-pipeline.ipynb
It uses all columns and have only one opreation for 'data cleaning'. Namaly the entries in column 'WheelTypeId' are converted to string:

```python
features_train.loc[:, 'WheelTypeID'] = features_train['WheelTypeID'].astype(str)
```

The main results are summariesd in the table. 
- f1_cv : f1 score obtained from cross-validation function during parameter selection on train data
- other scores are from sanity check on test data


| model | parameter | | f1_cv | | precision  | recall | f1-score | pred.sum() |
|-------|-----------|-|-------|-|------------|--------|----------|------------|
| Tree  | depth=4   |-| 0.376 |-| 0.30       | 0.49   | 0.37     | 2162       |
| Forest | depth=10 |-| 0.382 |-| 0.24       |  0.60  | 0.35     | 3143       |
| LogReg | C=0.01   |-| 0.375 |-| 0.19       | 0.80   | 0.30     | 5548       |
|Ensamble| -------- |-|------ |-| 0.32       | 0.48   | 0.38     | 1909       |

