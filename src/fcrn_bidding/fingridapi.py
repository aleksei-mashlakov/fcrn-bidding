import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from __main__ import logger_name

from utils import convert_to_strf, convert_to_utc

log = logging.getLogger(logger_name)


class FingridAPI(object):
    """https://data.fingrid.fi/en/pages/apis *publishes previous evening at 22:45 (EET)*
    Mapping:
        FCR-N hourly prices - "79"
        FCR-N hourly procured volumes - "80"
        FCR-N hourly activated reserve - "123"
        FCR-D hourly prices - "81"
        FCR-D hourly procured volumes - "82"
    """

    def __init__(self, config):
        self.variable_map = {
            "FCRN price": 79,
            "FCRN power": 80,
            "FCRN reserve activated": 123,
            "FCRD price": 81,
            "FCRD power": 82,
        }
        self.headers = {"Accept": config["Accept"], "x-api-key": config["x-api-key"]}

    def request_fingrid_data(
        self, type="FCRN price", request_period="last_day", return_local_time=False
    ):
        """Fetches the data using Fingrid API
        Args:
            type: variable type from the available at https://data.fingrid.fi/en/dataset
            start_time and end_time: in UTC local time format
        """
        if request_period == "last_day":
            start_time = datetime.utcnow().replace(
                hour=23, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
            end_time = start_time + timedelta(days=1)
        elif request_period == "next_day":
            start_time = datetime.utcnow().replace(
                hour=23, minute=0, second=0, microsecond=0
            )
            end_time = start_time + timedelta(days=1)

        start_time = convert_to_strf(start_time)
        end_time = convert_to_strf(end_time)

        params = (
            ("start_time", "{}".format(start_time)),
            ("end_time", "{}".format(end_time)),
        )

        log.info(
            "Requesting fingrid data for {} from {} to {} (UTC time)".format(
                type, start_time, end_time
            )
        )
        response = requests.get(
            "https://api.fingrid.fi/v1/variable/%s/events/csv"
            % (str(self.variable_map[type])),
            headers=self.headers,
            params=params,
        )

        resp = [line.decode("utf-8").split(",") for line in response.iter_lines()]
        df = pd.DataFrame(data=resp[1:], columns=resp[0]).drop("end_time", axis=1)
        df[["value"]] = df[["value"]].astype(float)
        df["start_time"] = pd.to_datetime(df["start_time"])
        df.set_index("start_time", inplace=True)
        df.index.rename("timestamp", inplace=True)
        # to convert to local time
        if return_local_time == True:
            df.index = (
                df.index.tz_localize("UTC")
                .tz_convert("Europe/Helsinki")
                .tz_localize(None)
            )
        return df
