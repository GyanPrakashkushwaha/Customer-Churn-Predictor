import os
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from churnPredictor import logger, CustomException
from churnPredictor.entity import DataTransformationConfig

class TransformData:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.config.train_data)
            test_df = pd.read_csv(self.config.test_data)

            logger.info("Read train and test data completed")

            X_train = train_df.drop(columns='Churn')
            y_train = train_df['Churn']
            X_test = test_df.drop(columns='Churn')
            y_test = test_df['Churn']

            # Manual encoding for binary feature
            X_train['Gender'] = X_train['Gender'].replace({'Male': 0, 'Female': 1})
            X_test['Gender'] = X_test['Gender'].replace({'Male': 0, 'Female': 1})

            # Define the transformer
            preprocessing = ColumnTransformer(transformers=[
                ('OHE', OneHotEncoder(drop='first', sparse_output=False, dtype=np.int64), ['Location']),
                ('scaling', MinMaxScaler(), ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
            ], remainder='passthrough')

            # Fit on Training data, Transform both
            transformed_train = preprocessing.fit_transform(X_train)
            transformed_test = preprocessing.transform(X_test)

            # Convert back to DataFrame for saving
            transformed_train_df = pd.DataFrame(data=transformed_train, columns=preprocessing.get_feature_names_out())
            transformed_test_df = pd.DataFrame(data=transformed_test, columns=preprocessing.get_feature_names_out())
            
            # Renaming columns to be cleaner (Optional but good for readability)
            rename_dict = {
                'OHE__Location_Houston': 'Houston',
                'OHE__Location_Los Angeles': 'LosAngeles',
                'OHE__Location_Miami': 'Miami',
                'OHE__Location_New York': 'NewYork',
                'scaling__Age': 'Age',
                'scaling__Subscription_Length_Months': 'Subscription_Length_Months',
                'scaling__Monthly_Bill': 'Monthly_Bill',
                'scaling__Total_Usage_GB': 'Total_Usage_GB',
                'remainder__Gender': 'Gender'
            }
            # Only rename if columns exist (safety check)
            transformed_train_df.rename(columns=rename_dict, inplace=True)
            transformed_test_df.rename(columns=rename_dict, inplace=True)

            # Save the files
            transformed_train_df.to_csv(self.config.transform_X_train_path, index=False)
            transformed_test_df.to_csv(self.config.transform_X_test_path, index=False)
            y_train.to_csv(self.config.y_train_path, index=False)
            y_test.to_csv(self.config.y_test_path, index=False)

            # Save the preprocessor object (CRITICAL for later use)
            joblib.dump(preprocessing, self.config.preprocessor_obj)

            logger.info("Data transformation completed and files saved!")
            logger.info(f"Transformed Data Shape: {transformed_train_df.shape}")

        except Exception as e:
            raise e