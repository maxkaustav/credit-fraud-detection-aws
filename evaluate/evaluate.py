import numpy as np
import pandas as pd
import os
import argparse

os.system("pip install mlflow")
import mlflow

def evaluate(model_uri):

    model = mlflow.tensorflow.load_model(model_uri)
    
    X_test = np.load("/opt/ml/processing/input/X_test.npy")
    y_test = np.load("/opt/ml/processing/input/y_test.npy")

    def predict_wrap(x):
        to_binary = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        return to_binary(np.squeeze(model.predict(x)))

    result = mlflow.models.evaluate(
    predict_wrap, # model
    X_test,
    targets=y_test,
    model_type="classifier",
    evaluators=None,
    evaluator_config={"log_explainer": False})
    
    print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"F1 Score: {result.metrics['f1_score']:.3f}")

if __name__ == '__main__':
    print("Started process.............")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uri', type=str, default='')

    args = parser.parse_args()

    evaluate(args.model_uri)
    
