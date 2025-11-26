from churnPredictor.constants import *
from churnPredictor.utils import read_yaml, create_dirs
from churnPredictor.entity import DataValidationConfig

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