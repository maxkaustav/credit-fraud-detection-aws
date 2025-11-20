
import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split

print("Reading data from loacation........")
data = pd.read_csv('/opt/ml/processing/input/creditcard.csv', delimiter=',')

feature_columns = data.columns[:-1]
label_column = data.columns[-1]

features = data[feature_columns].values.astype('float32')
labels = (data[label_column].values).astype('float32')

print(f"Features columns : {feature_columns}")
print(f"Labels columns : {label_column}")

print("Performing train-test split .............")

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=42)

print(f"Training length : {X_train.shape}")
print(f"Test length : {X_test.shape}")

print("Saving Transformed train data to: '/opt/ml/processing/output/train' ...")

np.save("/opt/ml/processing/output/X_train.npy", X_train)
np.save("/opt/ml/processing/output/y_train.npy", y_train)


print("Saving Transformed test data to: '/opt/ml/processing/output/test' ...")

np.save("/opt/ml/processing/output/X_test.npy", X_test)
np.save("/opt/ml/processing/output/y_test.npy", y_test)

print("completed processing ............")
