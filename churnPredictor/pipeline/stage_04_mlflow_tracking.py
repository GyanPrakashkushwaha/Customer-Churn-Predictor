from churnPredictor.configuration import ConfigurationManager
from churnPredictor.components.mlflow_tracking import TrackModelPerformance
from churnPredictor import logger

STAGE_NAME = "MLflow Tracking Stage"

class MLflowTrackingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            mlflow_tracking_config = config.get_mlflow_tracking_config()
            tracking = TrackModelPerformance(config=mlflow_tracking_config)
            tracking.start_mlflow()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MLflowTrackingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e