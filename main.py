# Put the code for your API here.
# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, load_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        os.system("dvc config core.hardlink_lock true")
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class FeatureConfig(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
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
        }


app = FastAPI()

model, encoder, lb ,sc= load_model(
    'model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl','model/scaler.pkl')




@app.get("/")
async def get_items():
    return {"message": "greeting"}


@app.post("/pred_values")
async def inference_main(input: FeatureConfig):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    input = input.dict(by_alias=True)
    # print("--------------------")
    # print(input)
    df = pd.DataFrame(data=input, index=[0])
    X_test, _, _, _ ,_= process_data(df, categorical_features=cat_features,
                                   training=False, label=None, encoder=encoder, lb=lb, scaler=sc)
    y_pred = inference(model, X_test)
    if y_pred[0]:
        pred = {"salary": ">50k"}
    else:
        pred = {"salary": "<=50k"}
    return pred