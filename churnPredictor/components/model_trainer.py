import os
import pandas as pd
import joblib
from churnPredictor import logger
from churnPredictor.entity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):
        try:
            # Load Data
            X_train = pd.read_csv(self.config.train_data)
            y_train = pd.read_csv(self.config.y_train_path)
            X_test = pd.read_csv(self.config.test_data)
            y_test = pd.read_csv(self.config.y_test_path)

            logger.info("Training data loaded successfully")

            # Define Models
            models = {
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "XGBoostClassifier": XGBClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "RandomForestClassifier": RandomForestClassifier()
            }

            # Train and Save Loop
            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                
                # Set parameters from params.yaml
                # We use .get(model_name) to be safe, in case a model is missing params
                specific_params = self.config.model_params_dir.get(model_name, {})
                model.set_params(**specific_params)

                # Train
                model.fit(X_train, y_train.values.ravel())

                # Save Model
                save_path = os.path.join(self.config.model_dir, f"{model_name}.joblib")
                joblib.dump(model, save_path)
                
                logger.info(f"{model_name} trained and saved at {save_path}")

        except Exception as e:
            raise e