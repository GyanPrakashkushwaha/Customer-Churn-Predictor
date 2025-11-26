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

## TODO
- read data from mondoDB 
- deploy the model in AWS
