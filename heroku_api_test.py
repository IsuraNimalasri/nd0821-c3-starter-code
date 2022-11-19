"""
Script for trying POST request on Heroku deployed APP
"""

import requests

body = {
    'age': 38,
    'workclass': 'Private',
    'fnlwgt': 215646,
    'education': 'HS-grad',
    'education-num': 9,
    'marital-status': 'Divorced',
    'occupation': 'Handlers-cleaners',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

resp = requests.post("https://udacity-project-03.herokuapp.com/pred_values", json=body)

print("Status code:", resp.status_code)
print("Response:", resp.json())