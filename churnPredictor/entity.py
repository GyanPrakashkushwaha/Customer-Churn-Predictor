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