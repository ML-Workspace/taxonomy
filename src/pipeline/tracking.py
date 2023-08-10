import mlflow
import torch
from src.logs import logger
from src.utility import Utils
import argparse
from datetime import datetime
from src.pipeline.evaluation import Evaluate
from transformers import pipeline
import os
from src.pipeline.training import TrainingPipeline
from src.pipeline.prediction import Predictions


class Tracking:
    def __init__(self, config_path):
        self.utils = Utils()
        self.config = self.utils.read_params(config_path)
        self.evaluation = Evaluate(config_path)

    def create_experiment(
        self, experiment_name, run_name, run_metrics, model_pipeline, args,run_params=None
    ):

        mlflow.set_tracking_uri("http://localhost:5000")

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):

            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])

            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])

            mlflow.set_tag("tag1", "Taxonomy Implementation")

            mlflow.pyfunc.log_model(artifacts={'pipeline': model_pipeline}, 
            artifact_path="Taxonomy_Model", python_model=Predictions(args.config))
        logger.info(
            "Run - %s is logged to Experiment - %s" % (run_name, experiment_name)
        )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    track = Tracking(config_path=parsed_args.config)

    run_params = track.config["estimators"]["params"]
    model_pipeline = os.path.join(os.getcwd(), track.config["estimators"]["model_dir"], "classifier")
    run_metrics = track.evaluation.model_evaluation()
    experiment_name = "Taxonomy_implementation" + str(
        datetime.now().strftime("%d-%m-%y")
    )
    run_name = "Taxonomy_implementation" + str(datetime.now().strftime("%d-%m-%y"))


    track.create_experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        run_metrics=run_metrics,
        model_pipeline=model_pipeline,
        args= parsed_args,
        run_params=run_params,
    )

