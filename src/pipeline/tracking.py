import mlflow
import mlflow.pytorch
import torch
from src.logs import logger
from src.utility import Utils
import argparse
from datetime import datetime
from src.pipeline.evaluation import Evaluate
from transformers import pipeline
import os
from src.pipeline.training import TrainingPipeline


class Tracking:
    def __init__(self, config_path):
        self.utils = Utils()
        self.evaluation = Evaluate()
        self.config = self.utils.read_params(config_path)

    def create_experiment(
        self, experiment_name, run_name, run_metrics, model, run_params=None
    ):

        mlflow.set_tracking_uri("http://localhost:5000")
        # use above line if you want to use any database like sqlite as backend storage for model else comment this line
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):

            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])

            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])

            mlflow.set_tag("tag1", "Taxonomy Implementation")
            mlflow.pytorch.log_model(model, "models")
        logger.info(
            "Run - %s is logged to Experiment - %s" % (run_name, experiment_name)
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    track = Tracking(config_path=parsed_args.config)
    run_params = track.config["estimators"]["params"]

    classifier = os.path.join(track.config["estimators"]["model_dir"], "taxonomy_model")
    model = pipeline("sentiment-analysis", model=classifier)

    run_metrics = track.evaluation.model_evaluation(config_path=parsed_args.config)
    experiment_name = "Taxonomy_implementation" + str(
        datetime.now().strftime("%d-%m-%y")
    )
    run_name = "Taxonomy_implementation" + str(datetime.now().strftime("%d-%m-%y"))


    # track.create_experiment(
    #     experiment_name=experiment_name,
    #     run_name=run_name,
    #     run_metrics=run_metrics,
    #     model=model,
    #     run_params=run_params,
    # )
    print(model)
    print(classifier)
