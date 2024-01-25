import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Set up logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")


# Reading the data
logging.info("Reading the data...")
df = pd.read_csv('bank.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df.drop(columns=['surname'],axis=1,inplace=True)
df.drop(columns=['id'],axis=1,inplace=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.exited.values
y_val = df_val.exited.values
y_test = df_test.exited.values

del df_train['exited']
del df_val['exited']
del df_test['exited']

categorical = list(df.dtypes[df.dtypes == 'object'].index)
numerical = list(df.select_dtypes(include=['int', 'float']).columns)
numerical.remove("exited")

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


xgb_params = {
    'eta': 0.01,
    'max_depth': 10,
    'min_child_weight': 13,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

# Exporting the model
logging.info("Exporting the model...")
with open('model.bin', 'wb') as f:
    pickle.dump((model, dv), f)