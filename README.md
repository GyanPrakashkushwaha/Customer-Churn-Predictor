# Customer-Churn-Predictor

## Data Insights:
- #### Numerical Data Points are equally distributed.
<img src="artifacts/readme/EDA.png" alt="EDA Image" width="700" >

<!-- ![EDA Image](artifacts/readme/EDA.png) -->
- #### Almost equally distribution of each class in each feature.
<img src="artifacts/readme/categorical_features.png" alt="EDA Image" width="700" >

<!-- ![Cat Features](artifacts/readme/categorical_features.png) -->
- #### With respect to churn (equal contribution of all classes)
<img src="artifacts/readme/output.png" alt="EDA Image" width="700" >
<!-- ![with repect to churn](artifacts/readme/output.png) -->


### Installation
1. Clone the repository and navigate:
   
```python
git clone https://github.com/GyanPrakashkushwaha/Customer-Churn-Prediction.git customer-churn-prediction ; cd customer-churn-prediction
```

2. Create virtaul environment and activate it.
```python
virtalenv churnvenv 
churnvenv/Scipts/activate.ps1
```

3. Install the required dependencies:
```python
pip install -r requirements.txt
```
4. run main.py for `data validation` , `data transformation`, `model training` and `mlflow tracking`.
```Python
python run main.py
```  

5. Run the the streamlit app:
```python
streamlit run app.py
```

### MLflow 

- MLflow for local web server
```Python
mlflow ui
```

- run this in environment 
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow
export MLFLOW_TRACKING_USERNAME=GyanPrakashKushwaha 
export MLFLOW_TRACKING_PASSWORD=53950624aa84e08b2bd1dfb3c0778ff66c4e7d05
```
- Tracking URL
```Python
https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow
```

### **I TRIED MY BEST!** 😓
- For **model performance Improvement**(Data manipulation) `normalized` the features using `log normal distribution` but the performance didn't increase and then tried `Generated Data`  using `SMOTE` and then `trained model` in the large data but still the `accuracy` remained `same`.
- For **model performance Improvement** (Model training) Used `complex Algorithms` - `GradientBoostingClassifier` , `XGBoostClassifier` , `CatBoostClassifier` , `AdaBoostClassifier` , `RandomForestClassifier`
to easy algorithm like `Logistic Regession` and Also trained `Deep Neural Network` with different `weight Initializers` , `activation function` ,`input nodes` and `optimizer` but **models performance not Improved** .


## TODO
- read data from mondoDB 
- deploy the model in AWS
