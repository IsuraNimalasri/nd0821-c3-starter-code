import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def load_model(model_path, encoder_path, lb_path,sc_path):
    """
    load the saved model
    """
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_encoder = pickle.load(open(encoder_path, 'rb'))
    loaded_lb = pickle.load(open(lb_path, 'rb'))
    loaded_sc = pickle.load(open(sc_path, 'rb'))
    return loaded_model, loaded_encoder, loaded_lb ,loaded_sc


def slice_evaluation(df, model, cat_features, encoder, lb,sc):
    """
    computes performance on model slices.

    Inputs
    ----------
    df: test dataframe
    model: trained model
    cat_features: categorical features
    encoder: encoded dataframe
    lb: encoded label
    Returns
    -------
    """
    with open("model/slice_output.txt", "w") as f:
        for feature in cat_features:
            for cls in df[feature].unique():
                slice = df[df[feature] == cls]
                x_test, y_test, _, _ ,_= process_data(
                    slice,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb,scaler=sc)
                y_pred_slice = inference(model, x_test)
                precision, recall, fbeta = compute_model_metrics(y_test, y_pred_slice)
                slice_metric = f"cat: {feature:}, var: {cls}, precision {precision:.2f}, recall {recall:.2f}, F1 {fbeta:.2f} \n"
                f.write(slice_metric)

