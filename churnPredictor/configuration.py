from churnPredictor.constants import *
from churnPredictor.utils import read_yaml, create_dirs
from churnPredictor.entity import (DataValidationConfig, 
                                   DataTransformationConfig, 
                                   ModelTrainerConfig, 
                                   MLFlowTrackingConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_dirs([self.config.artifacts_root])
        
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.INDENPENT_FEATURES
        
        create_dirs([config.root_dir])
        
        return DataValidationConfig(
            root_dir= config.root_dir,
            data_dir= config.data_dir,
            STATUS_FILE=config.STATUS_FILE,
            make_data= config.make_data,
            all_schema= schema
        )
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_to_train_model

        create_dirs([config.root_dir, config.model_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data=config.train_data_path,
            test_data=config.test_data_path,
            transform_X_test_path=config.transform_X_test_path,
            transform_X_train_path=config.transform_X_train_path,
            y_test_path=config.y_test_path,
            y_train_path=config.y_train_path,
            preprocessor_obj=config.preprocessor_obj,
            model=config.model_dir,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.models  

        create_dirs([config.model_dir])

        model_trainer_config = ModelTrainerConfig(
            train_data=config.train_data,
            test_data=config.test_data,
            model_dir=config.model_dir,
            y_train_path=config.y_train_path,
            y_test_path=config.y_test_path,
            model_params_dir=params
        )

        return model_trainer_config
    
    def get_mlflow_tracking_config(self) -> MLFlowTrackingConfig:
        config = self.config.mlflow_tracking

        create_dirs([config.mlflow_dir])

        mlflow_tracking_config = MLFlowTrackingConfig(
            root_dir=config.mlflow_dir,
            test_data=config.test_data,
            model_dir=config.model_dir,  # Pointing to where models are saved
            metrics_file=config.metrics_file ,
            confusion_matrix_image=config.confusion_metrics,
            y_test_path=config.y_test_path,
            mlflow_uri= config.mlflow_uri
        )

        return mlflow_tracking_config