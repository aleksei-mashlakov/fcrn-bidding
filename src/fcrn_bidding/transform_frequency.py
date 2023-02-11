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
        self.Proxy.transform_frequency(request_period="last_two_months")
        self.Proxy.transform_frequency(request_period="two_months_before_two_months")
        self.Proxy.transform_frequency(request_period="two_months_before_four_months")


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
