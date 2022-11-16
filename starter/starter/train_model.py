# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
from starter.starter.ml import data, model
import pickle
import csv

# Add code to load in the data.
def import_data(path):
    '''
    :param path: path to csv file
    :return: df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df

def save_model(path, t_model):
    '''
    Saving trained model to specified path dir
    :param path: path to model dir
    :param t_model: trained model
    '''
    with open(path, 'wb') as m_file:
        pickle.dump(t_model, m_file)
        
def load_model(model_path, encoder_path, lb_path):
    """
    load the saved model
    """
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_encoder = pickle.load(open(encoder_path, 'rb'))
    loaded_lb = pickle.load(open(lb_path, 'rb'))
    return loaded_model, loaded_encoder, loaded_lb
        
def slicing_perfo(path, features_lst, df, enc, binarizer, mdl):
    '''
    Slicing and model performances
    :param path: path to log dir
    :param features_lst: categorical feature list
    :param df: dataframe
    :param enc: encoder
    :param binarizer: binarizer
    :param mdl: trained model
    '''

    file_exists = os.path.isfile(path)
    for feature in features_lst:
        for feature_value in df[feature].unique():
            # slicing dataframe with respect to feature and feature_value
            sliced_df = df[df[feature] == feature_value]
            X_slice, y_slice, _, _ = data.process_data(
                sliced_df,
                categorical_features=features_lst,
                label="salary", training=False,
                encoder=enc, lb=binarizer)
            predictions_slice = model.inference(mdl, X_slice)
            precision_sl, recall_sl, f_beta_sl = model.compute_model_metrics(y_slice, predictions_slice)
            with open(path, 'a') as csvfile:
                fieldnames = ['feature', 'feature_value', 'precision', 'recall', 'f_beta']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()  # writing header only if file does not exist
                    file_exists = True

                writer.writerow({'feature': feature,
                                 'feature_value': feature_value,
                                 'precision': precision_sl,
                                 'recall': recall_sl,
                                 'f_beta': f_beta_sl})
      
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
def train(X_train, y_train):
    _model = model.train_model(X_train, y_train)
    with open("model/model.pkl", 'wb') as file_model:
        pickle.dump(_model, file_model)
    return _model

def test_model(test, cat_features=None, label="salary"):
    # load model
    model, encoder, lb = load_model(
        'model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl')
    # testing
    X_test, y_test, _, _ = data.process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb)

    preds = model.inference(model, X_test)
    precision, recall, fbeta = model.compute_model_metrics(y_test, preds)
    return precision, recall, fbeta