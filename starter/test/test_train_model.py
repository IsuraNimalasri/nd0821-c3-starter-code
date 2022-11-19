import os
from starter.train_model import preprocessing, train

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

def test_preprocessing():
    _, _, encoder, lb, _ = preprocessing("data/census_cleaned_copy.csv", cat_features)
    assert os.path.isfile("model/encoder.pkl")
    assert os.path.isfile("model/lb.pkl")

def test_train():
    X_train, y_train, _, _, _ = preprocessing("data/census_cleaned_copy.csv", cat_features)
    train(X_train, y_train)
    assert os.path.isfile("model/model.pkl")