from starter.ml.model import slice_evaluation
from starter.train_model import preprocessing, train, test_model


cat_features = [
"workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country"]

X_train, y_train, encoder, lb, df_test = preprocessing("data/census_cleaned_copy.csv", cat_features)
model = train(X_train, y_train)
p,r,f = test_model(df_test, cat_features, label="salary" )
print(p,r,f)
slice_evaluation(df_test, model, cat_features, encoder, lb)