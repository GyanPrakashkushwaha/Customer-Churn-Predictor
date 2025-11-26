from churnPredictor.constants import *
from churnPredictor.utils import read_yaml, create_dirs
from churnPredictor.entity import DataValidationConfig, DataTransformationConfig

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