# Customer-Churn-Predictor


## problems faced
- each time I have to build the complete project using setup.py
- scikit-learn expects 1D array(with no columns), but the data frame gives (row, 1), I was pretty frustrated but evantually used y_train.values.ravel() to resolve that.