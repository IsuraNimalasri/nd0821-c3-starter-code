import os
import pytest
import pandas as pd
from starter.train_model import preprocessing, train
from starter.ml.model import train_model, compute_model_metrics, inference


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]


@pytest.fixture
def fake_data():
    """
    Fake data for model training , this dataframe use to test
    """
    df = pd.DataFrame({
        "var1": [1, 2, -3, -1, 2, 3],
        "var2": [0, 0, 0, 1, 1, 1],
        "var3": [2.7, 1.5, -0.8, 0.2, -2, 0.3],
        "label": [1, 1, 1, 1, 0, 0]
    })

    return df


def test_preprocessing():
    _, _, encoder, lb, _, _ = preprocessing(
        "data/census_cleaned.csv", cat_features)
    assert os.path.isfile("model/encoder.pkl")
    assert os.path.isfile("model/lb.pkl")


def test_train():
    X_train, y_train, _, _, _, _ = preprocessing(
        "data/census_cleaned_copy.csv", cat_features)
    train(X_train, y_train)
    assert os.path.isfile("model/model.pkl")


def test_train_model(fake_data):
    """
    Tests if a model can be correctly trained
    """
    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")

    model = train_model(X_fake, y_fake)

    assert model.n_classes_ == 2


def test_inference(fake_data):

    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")
    model = train_model(X_fake, y_fake)

    assert all(inference(model, X_fake) == [1, 1, 1, 1, 0, 0])
