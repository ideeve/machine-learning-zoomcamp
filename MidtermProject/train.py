import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



# Set up logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# Reading the data
logging.info("Reading the data...")
df = pd.read_csv('Breast_Cancer.csv')

# Data cleaning


df["Status"] = df["Status"].replace(["Alive", "Dead"], [1, 0])

df.columns = df.columns.str.lower().str.replace(' ', '_')

# Splitting the data 60:20:20
logging.info("Splitting the data...")

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.status.values
y_val = df_val.status.values
y_test = df_test.status.values

del df_train['status']
del df_val['status']
del df_test['status']

categorical = list(df.dtypes[df.dtypes == 'object'].index)
numerical = list(df.dtypes[df.dtypes == 'int'].index)
numerical.remove("status")


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

rf = RandomForestClassifier(n_estimators=50,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)

# Exporting the model
logging.info("Exporting the model...")
with open('model.bin', 'wb') as f:
    pickle.dump((rf, dv), f)