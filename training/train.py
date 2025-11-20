
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import keras
import numpy as np

import mlflow


def get_data(train_dir):
    trainx , trainy = np.load(f"{train_dir}/X_train.npy"),np.load(f"{train_dir}/y_train.npy")
    return trainx, trainy

def get_model(trainx):
    
    model = keras.Sequential(
    [
        keras.Input(shape=trainx.shape[1:]),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    return model


def train_model(model,x,y,epochs):
    
    metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
    )

    callbacks = []

    model.fit(
        x,
        y,
        batch_size=2048,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_split=0.1,
    )

    return model
    

if __name__ == '__main__':
    
    print("Started process.............")
    parser = argparse.ArgumentParser()
    
    print(os.listdir("/opt/ml/input/data/train"))
    
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--epochs', type=str, default=10)
    
    args = parser.parse_args()
    x, y = get_data(train_dir)
    
    print(f"x train shape : {x.shape}")
    print(f"y train shape : {y.shape}")
    
    
    # # Set the tracking server URI using the ARN of the tracking server you created
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
    
    # # Enable autologging in MLflow
    mlflow.autolog()

    print("Building model..........")
    model = get_model(x)

    print(model.summary())

    print("Training model..........")

    trained_model = train_model(model,x,y,int(args.epochs))

    print("Completed training........")
    

    
