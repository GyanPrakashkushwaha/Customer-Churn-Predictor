import os
import pandas as pd
from churnPredictor import logger
from churnPredictor.entity import DataValidationConfig
from sklearn.model_selection import train_test_split

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)
            
            for col in all_cols:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f'Validation Status: {validation_status}')
        except:
            validation_status = True
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_dir)
        train_data, test_data = train_test_split(data, test_size=0.25)
        
        train_data.to_csv(self.config.make_data.train_data, index=False)
        test_data.to_csv(self.config.make_data.test_data, index=False)

        logger.info(f"Splited data into training and test sets")
        logger.info(f"train_data shape: {train_data.shape}")
        logger.info(f"test_data shape: {test_data.shape}")