"""
Prediction method based on gluonts library
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

### Gluonts ###
import mxnet as mx

### Common imports ###
import numpy as np
import pandas as pd
from mxnet import gluon
from pandas.tseries.frequencies import to_offset
from tqdm import tqdm

mx.random.seed(0)
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

DISTRIBUTIONS = {
    "gaussian": "GaussianOutput",
    "student_t": "StudentTOutput",
    "neg_binomial": "NegativeBinomialOutput",
}

np.random.seed(0)


### Logging ###
import logging

from __main__ import logger_name

log = logging.getLogger(logger_name)


class Predictor(object):
    """A wrapper around glounts [https://github.com/awslabs/gluon-ts] library
    for probabilistic timeseries forecasting"""

    def __init__(self, config):
        self.save_folder = "./data/training"
        self.plot_log_path = "./plots/"
        self.map_model_to_targets = dict()
        self.config = config["models"]
        self.models = {k: self.init_model(k, v) for k, v in config["models"].items()}
        self.folders = self.init_folders()
        log.info(f"Predictor.map_model_to_targets: {self.map_model_to_targets}")

    def get_distribution(self, submodule):
        import importlib

        module = importlib.import_module(f"gluonts.distribution.{submodule}")
        class_ = getattr(module, DISTRIBUTIONS[submodule])
        instance = class_()
        return instance

    def init_folders(self, folders=dict()):
        for predictor_name in next(os.walk(self.save_folder))[1]:
            folders[predictor_name] = os.path.join(self.save_folder, predictor_name)
        return folders

    def init_model(self, model_name, config):
        """Initializes the prediction model"""
        log.info(f"Initializing the {model_name} estimator")
        if config["type"] == "DeepAR":
            estimator = self.init_DeepAR(config)
        elif config["type"] == "FFNN":
            estimator = self.init_FFNN(config)
        elif config["type"] == "Naive":
            estimator = None
        else:
            log.error(f"Estimator {model_name} type is not specified")
        self.map_model_to_targets[model_name] = config["targets"]
        return estimator

    def init_FFNN(self, config):
        """Initializes the FFNN estimator with given configuration"""
        estimator = SimpleFeedForwardEstimator(
            prediction_length=config["horizon"],
            freq=config["freq"],
            context_length=config["window"],
            distr_output=self.get_distribution(config["output"]),
            trainer=Trainer(
                ctx="cpu",
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                num_batches_per_epoch=config["num_batches_per_epoch"],
                batch_size=config["batch_size"],
                patience=config["patience"],
            ),
        )
        return estimator

    def naive_weekly_seasonal(self, y_true_df, horizon):
        """NOTE: y_true_df with zeros for the horizon"""
        h = 1
        for i, row in y_true_df.iloc[-horizon:].iterrows():
            if h <= 96:
                y_true_df.loc[i, :] = y_true_df.loc[
                    (i - timedelta(minutes=96 * 15)), :
                ].values
            else:
                y_true_df.loc[i, :] = y_true_df.loc[
                    (i - timedelta(minutes=(192 * 15))), :
                ].values
            h += 1
        return y_true_df.iloc[-horizon:]

    def init_DeepAR(self, config):
        """Initializes the DeepAR estimator with given configuration"""
        estimator = DeepAREstimator(
            prediction_length=config["horizon"],
            freq=config["freq"],
            context_length=config["window"],
            distr_output=self.get_distribution(config["output"]),
            use_feat_dynamic_real=True,
            dropout_rate=config["dropout_rate"],
            cell_type=config["rnn_cell"],
            trainer=Trainer(
                ctx="cpu",
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                num_batches_per_epoch=config["num_batches_per_epoch"],
                batch_size=config["batch_size"],
                patience=config["patience"],
            ),
        )
        return estimator

    def train(self, dfs_dict: dict, trained=False):
        """Trains and saves the prediction model
        Args:
            - dict of dataframes with history time series (UTC index)
        Examples:
            >>> Predictor.train(df_dict)
        """
        if dfs_dict is None or not isinstance(dfs_dict, dict):
            log.error(f"Function argument must be a valid dict")

        dfs = pd.concat(list(dfs_dict.values()), axis=1, ignore_index=False)
        log.debug(f"Training dataframe: {dfs}")
        for model_name, targets in self.map_model_to_targets.items():
            if dfs.columns.isin(targets).all():
                log.info(f"Found {model_name} estimator for targets: {targets}")
                estimator = self.models[model_name]
                df = dfs[dfs.columns]
                log.debug(f"Obtaining train and test ListDatasets ...")
                (freq, horizon, window) = self.get_params(model_name)
                train_ds, test_ds = self.create_list_dataset(
                    df, horizon=horizon, freq=freq, submission=False
                )
                log.info(f"Start training {model_name} estimator ... ")
                predictor = estimator.train(train_ds)
                trained = True
                predictor_name = self.get_folder(model_name, dfs.columns).split("/")[
                    -2
                ]  # .split('_')[0]
                self.folders[predictor_name] = self.get_folder(model_name, dfs.columns)
                self.save_model(predictor_name, predictor)
        if trained:
            log.info(
                f"Predicting targets {targets} with {predictor_name} predictor ... "
            )
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_ds, predictor=predictor, num_samples=100
            )
            log.debug("Obtaining time series conditioning values ...")
            tss = list(tqdm(ts_it, total=df.shape[1]))
            log.debug("Obtaining time series predictions ...")
            forecasts = list(tqdm(forecast_it, total=df.shape[1]))
            self.evaluate(tss, forecasts)
            # log.info("Plotting time series predictions ...")
            # self.plot_forecasts(tss, forecasts, window, horizon)
        else:
            log.error(f"No estimators found for the targets: {targets}")
        return

    def predict(self, dfs_dict: dict, submission=True) -> dict:
        """
        Makes predictions in future
        Args:
            - dict of dataframes with history time series (UTC index)
        Returns:
            - dict of dataframes with predicted time series (UTC index)
        Examples:
            >>> Predictor.predict(dfs_dict)
        """
        if dfs_dict is None or not isinstance(dfs_dict, dict):
            log.error("Function argument must be a valid dict")
        else:
            dfs = pd.concat(list(dfs_dict.values()), axis=1, ignore_index=False)

        log.debug(f"Forecasting dataframe: {dfs}")
        for model_name, targets in self.map_model_to_targets.items():
            if dfs.columns.isin(targets).all():
                log.info(f"Found {model_name} estimator for targets {targets}")
                df = dfs[dfs.columns]
                predictor_name = self.get_folder(model_name, dfs.columns).split("/")[
                    -2
                ]  # .split('_')[0]
                if not (
                    predictor_name in self.folders
                ):  # or predictor_name in model_folders):
                    log.warning(
                        f"No saved predictors found for {predictor_name} predictor. Re-training."
                    )
                    self.train(dfs_dict)
                predictor = self.load_model(predictor_name)
                log.debug("Obtaining test ListDataset ...")
                (freq, horizon, window) = self.get_params(model_name)
                _, test_ds = self.create_list_dataset(
                    df, horizon=horizon, freq=freq, test=True, submission=True
                )
                log.info(
                    f"Predicting targets {targets} with {predictor_name} predictor"
                )
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=test_ds, predictor=predictor, num_samples=100
                )
                log.debug("Obtaining time series conditioning values ...")
                tss = list(tqdm(ts_it, total=df.shape[1]))
                log.debug("Obtaining time series predictions ...")
                forecasts = list(tqdm(forecast_it, total=df.shape[1]))
                # log.info("Plotting time series predictions ...")
                # self.plot_forecasts(tss, forecasts, window, horizon)

                if submission:
                    df_dict = dict()
                    start = df.index[-1] + pd.to_timedelta(to_offset(freq))
                    for i in range(len(forecasts)):
                        log.info(
                            f"Saving prediction dataframes for {dfs.columns[i]} forecast"
                        )
                        df_frcst = pd.DataFrame(
                            index=pd.date_range(start=start, periods=horizon, freq=freq)
                        )
                        for quantile in ["q25", "q50", "q75"]:
                            df_frcst[quantile] = forecasts[i].quantile(
                                float(quantile[-2:]) * 0.01
                            )
                        df_dict[dfs.columns[i]] = df_frcst
                    return df_dict
                else:
                    return forecasts

    def evaluate(self, tss, forecasts):
        """Evaluates the model performance with common metrics"""
        evaluator = Evaluator(quantiles=[0.25, 0.5, 0.75])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
        log.info(
            f"Predictor evaluation metrics: \n {json.dumps(agg_metrics, indent=4)}"
        )
        return

    def create_list_dataset(
        self, df, horizon=24, freq="H", test=False, submission=True
    ) -> ListDataset:
        """Creates the list dataset for training and testing"""
        train_target_values = df.T.values
        dates = [
            pd.Timestamp(df.index[0], freq=freq)
            for _ in range(train_target_values.shape[0])
        ]

        if submission == True:
            test_target_values = [
                np.append(ts, np.ones(horizon) * np.nan) for ts in train_target_values
            ]
            feat_dynamic_real_train = [
                [np.append(np.ones(horizon) * 0, ts[:-horizon])]
                for ts in train_target_values
            ]
            feat_dynamic_real_test = [
                [np.append(np.ones(horizon) * 0, ts[:])] for ts in train_target_values
            ]
        else:
            test_target_values = train_target_values.copy()
            feat_dynamic_real_train = [
                [np.append(np.ones(horizon) * 0, ts[: -2 * horizon])]
                for ts in train_target_values
            ]
            feat_dynamic_real_test = [
                [np.append(np.ones(horizon) * 0, ts[:-horizon])]
                for ts in train_target_values
            ]
            train_target_values = [ts[:-horizon] for ts in train_target_values]

        feat_static = [[x] for x in range(df.T.values.shape[1])]

        train_ds = ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_DYNAMIC_REAL: fdr,
                    FieldName.FEAT_STATIC_CAT: fsc,
                }
                for (target, start, fdr, fsc) in zip(
                    train_target_values, dates, feat_dynamic_real_train, feat_static
                )
            ],
            freq=freq,
        )

        test_ds = ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_DYNAMIC_REAL: fdr,
                    FieldName.FEAT_STATIC_CAT: fsc,
                }
                for (target, start, fdr, fsc) in zip(
                    test_target_values, dates, feat_dynamic_real_test, feat_static
                )
            ],
            freq=freq,
        )
        return train_ds, test_ds

    def get_folder(self, model_name, target_list) -> str:
        """Generates folder name based on target list"""
        t_list = [target.replace(" ", "_") for target in target_list]
        dir_name = os.path.join(
            self.save_folder, model_name + "_" + "_".join(t_list) + "/"
        )
        directory = os.path.dirname(dir_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return dir_name

    def load_model(self, predictor_name):
        """Loads the model back from the folder"""
        from gluonts.model.predictor import Predictor

        return Predictor.deserialize(Path(self.folders[predictor_name]))

    def save_model(self, predictor_name, predictor):
        """Saves the trained model to the folder"""
        return predictor.serialize(Path(self.folders[predictor_name]))

    def get_params(self, model_name):
        """Retrieves the parameter from config file"""
        params = self.config[model_name]
        return params["freq"], params["horizon"], params["window"]

    def plot_forecasts(self, tss, forecasts, window, horizon):
        """Plots and saves all the forecasted series"""
        directory = os.path.dirname(self.plot_log_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in tqdm(range(len(forecasts))):
            ts_entry = tss[i]
            forecast_entry = forecasts[i]
            plot_prob_forecasts(
                ts_entry,
                forecast_entry,
                self.plot_log_path,
                i,
                window,
                horizon,
                inline=False,
            )
        return


def plot_prob_forecasts(
    ts_entry,
    forecast_entry,
    path,
    sample_id,
    plot_length,
    prediction_length,
    inline=False,
):
    """Plots probabilistic forecasts"""
    prediction_intervals = (50, 67)  # , 95, 99)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
    ax.axvline(ts_entry.index[-prediction_length], color="r")
    plt.legend(legend, loc="upper left")
    if inline:
        plt.show()
        plt.clf()
    else:
        plt.savefig("{}forecast_{}.pdf".format(path, sample_id))
        plt.close()
    return
