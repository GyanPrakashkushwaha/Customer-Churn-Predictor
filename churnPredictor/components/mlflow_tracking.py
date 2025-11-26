import os
import json
import joblib
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from churnPredictor import logger
from churnPredictor.entity import MLFlowTrackingConfig

class TrackModelPerformance:
    def __init__(self, config: MLFlowTrackingConfig):
        self.config = config

    def evaluate(self, true, pred, model_name):
        # Calculate metrics
        accuracy = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)

        # Create and Save Confusion Matrix
        cm = confusion_matrix(true, pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(data=cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        
        # Save image
        cm_path = os.path.join(self.config.root_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        evaluation_report = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        }
        return evaluation_report

    def start_mlflow(self):
        try:
            # 1. Setup MLflow
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            logger.info(f"MLflow tracking URI set to: {self.config.mlflow_uri}")

            # 2. Load Test Data
            X_test = pd.read_csv(self.config.test_data)
            y_test = pd.read_csv(self.config.y_test_path)

            # 3. Find all trained models
            model_dir = self.config.model_dir
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]

            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                model_path = os.path.join(model_dir, model_file)
                
                # Load Model
                model = joblib.load(model_path)
                logger.info(f"Evaluating model: {model_name}")

                # Start MLflow Run
                mlflow.set_experiment("Customer_Churn_Prediction")
                with mlflow.start_run(run_name=model_name):
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Evaluate
                    metrics = self.evaluate(y_test, y_pred, model_name)
                    
                    # Log Metrics to MLflow
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("model_name", model_name)
                    
                    # Log Model to MLflow
                    mlflow.sklearn.log_model(model, "model")
                    
                    # Log Confusion Matrix Image
                    cm_path = os.path.join(self.config.root_dir, f"{model_name}_confusion_matrix.png")
                    mlflow.log_artifact(cm_path)

                    # Save local JSON metrics
                    with open(self.config.metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=4)

            logger.info("MLflow tracking completed for all models.")

        except Exception as e:
            raise e