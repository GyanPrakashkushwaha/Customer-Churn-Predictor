from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: Path
    all_schema: dict
    make_data: dict
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data: Path
    test_data: Path
    transform_X_train_path: Path
    y_train_path: Path
    transform_X_test_path: Path
    y_test_path: Path
    preprocessor_obj: Path
    model: Path
    
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    train_data: Path
    test_data: Path
    model_dir: Path
    y_train_path: Path
    y_test_path: Path
    model_params_dir: dict
    
    
@dataclass(frozen=True)
class MLFlowTrackingConfig:
    root_dir: Path
    test_data: Path
    model_dir: Path
    metrics_file: Path
    confusion_matrix_image: Path
    y_test_path: Path
    mlflow_uri: str