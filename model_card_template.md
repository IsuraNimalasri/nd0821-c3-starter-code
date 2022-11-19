# Model Card

## Model Details

This is a Random Forest Classifier from Scikit Learn created by Isura Nimalasiri for the final project of the third course of the MLOps nanodegree at Udacity. 

## Intended Use

This model should be used to predict if a USA citizen makes or not more than 50K$ per year provided demographic and financial characteristics. Note however that since the data was collected in 1994 it should not be used for predicting on current demographic data.

## Training Data

- Data comes from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original set contains 32561 rows.

-  All categorical variables were One-Hot encoded, while continuous variables were scaled with a Standar Scaler for better Random Forest performance.

-  The label column "salary" was also transformed by Label Binarizer.

## Evaluation Data

Metrics were calculated on each 20% size validation sample and then averaged.

## Metrics

precision : 0.7296105
recall:0.628083491
fbeta: 0.675050

## Ethical Considerations

The model was also trained and validated on an 80-20 split in order to check performance on data slices according to each categorical value.

Some data slices showed a performance much lower than the model average. We present here a couple of examples:

Precision is much lower (less than 50%) for a handful of Native-Country values (Jamaica, South, Mexico and China).
The highest False Positive Rate occurs mainly on four levels of higher education (Doctorate, Prof-School, Masters and Bachelors).
The given data's feature imbalance produces very high variation between slices. It should be used with care depending on the intended goal.



## Caveats and Recommendations

This model was trained with census data from 1994. Though it provides a useful experiment, no conclusion should be drawn from applying this model to current data, more than 25 years later. It is better to see this model as a way to obtain insights from the home economy landscape of the 90s.