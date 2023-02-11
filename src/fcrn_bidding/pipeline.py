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
    """
    Class responsible for pipeline fuctionality
    """

    def __init__(self, config):
        self.active = False
        self.Proxy = Proxy(config)
        self.run()

    def run_threaded(self, job_func):
        """
        Running job on thread
        """
        job_thread = threading.Thread(target=job_func)
        job_thread.start()

    def run(self):
        """
        Runs the main processing pipeline
        """
        log.info("Start running the pipeline")
        # self.Proxy.update_fingrid_api()
        # self.Proxy.transform_frequency(request_period="last_six_months")
        # self.Proxy.train_models()
        # log.info(f"{self.forecast_load(request_period="last_year")}")
        self.Proxy.submit_bids()
        # self.Proxy.update_bids()
        # self.Device.update_state()
        self.active = False

        if self.active == True:
            # Fingrid prices are stated to be published at 22:45
            # schedule.every().day.at("02:10").do(self.run_threaded, self.Proxy.update_fingrid_api)
            # schedule.every().sunday.at("16:05").do(lambda s=self: s.Proxy.train_models())
            schedule.every(2).to(3).minutes.do(self.run_threaded, self.Proxy.keep_alive)
            schedule.every().day.at("09:20").do(
                self.run_threaded, self.Proxy.submit_bids
            )
            schedule.every().day.at("21:59").do(
                self.run_threaded, self.Proxy.update_bids
            )
            for minutes in range(0, 60, 15):
                schedule.every().hour.at(":%02d" % (minutes)).do(
                    self.run_threaded, self.Proxy.verify_the_job
                )
            for minutes in [13, 28, 43, 58]:
                schedule.every().hour.at(":%02d" % (minutes)).do(
                    self.run_threaded, self.Proxy.update_state
                )

        while self.active:
            schedule.run_pending()
            time.sleep(1)
        # self.stop()

    def stop(self):
        """
        Emergency/testing interruption
        """
        log.warning(f"Emergency interruption. Stopping the scheduler.")
        self.Proxy.Device.deactivate()
        self.active = False
        return


if __name__ == "__main__":
    config_file = "./configuration.yml"
    config = yaml.full_load(open(config_file, "r"))
    log = LogInit(
        logger_name,
        os.path.join(".", "logs", "logs.log"),
        debuglevel=log_level(config["logLevel"]),
        log=True,
    )
    log.info(os.getcwd())
    log.info(f"Python version: {sys.version}")
    pp = Pipeline(config=config)
