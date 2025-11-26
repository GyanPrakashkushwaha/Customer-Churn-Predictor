import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

class PredictionPipeline:
    def __init__(self):
        
        self.preprocessor = joblib.load(Path('artifacts/preprocessor/preprocessorObj.joblib'))
        self.model = joblib.load(Path(r'artifacts\model\XGBoostClassifier.joblib'))

    def predict(self, data: pd.DataFrame):
        
        # 1. Manual Encoding for Gender (Matches what we did in Data Transformation)
        # We check if values are strings 'Male'/'Female' before replacing, just in case
        if 'Gender' in data.columns:
            data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
        
        # 2. Apply the Preprocessor (Scaling + OneHotEncoding)
        data_transformed = self.preprocessor.transform(data)
        
        # 3. Make Prediction
        prediction = self.model.predict(data_transformed)
        
        return prediction