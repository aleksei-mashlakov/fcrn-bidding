"""
This module should contain your main project pipeline(s).

Whilst the pipeline may change during the analysis phases, any more stable pipeline should be implemented here so
that it can be reused and easily reproduced.
"""
# This must be set in the beggining because in model_util, we import it
logger_name = "FCRN-BID"
import logging
import os
import sys
import threading
import time

import schedule
import yaml

from MsgLog import LogInit
from proxy import Proxy
from utils import log_level


class Pipeline(object):
    """Class responsible for pipeline fuctionality"""

    def __init__(self, config):
        self.active = False
        self.Proxy = Proxy(config)
        self.run()

    def run(self):
        """Runs the main processing pipeline"""
        log.info("Start running the pipeline")
        self.Proxy.Predictor.train(
            self.Proxy.get_fcrn_price_time_series(request_period="last_three_years")
        )
        self.Proxy.Predictor.train(
            self.Proxy.get_wholesale_price_time_series(
                request_period="last_three_years"
            )
        )
        # self.Proxy.Predictor.train(self.Proxy.get_bess_fcrn_time_series(request_period='last_five_years'))
        # self.Proxy.Predictor.train(self.Proxy.get_load_time_series({'lut': ['total']},
        #                                                             scale=0.2,
        #                                                             request_period='last_three_years'))
        # self.Proxy.Predictor.train(self.Proxy.get_load_time_series({'LVDC-load': ['amr_kw']},
        #                                                             offset='3Y',
        #                                                             request_period='last_year'))
        # self.Proxy.train_models()
        # self.Proxy.submit_bids()
        # self.Proxy.update_bids()


if __name__ == "__main__":
    config_file = "./configuration.yml"
    config = yaml.full_load(open(config_file, "r"))
    log = LogInit(
        logger_name,
        os.path.join(".", "logs", "logs.log"),
        log_level(config["logLevel"]),
        True,
    )
    log.info(os.getcwd())
    log.info(f"Python version: {sys.version}")
    pp = Pipeline(config=config)
