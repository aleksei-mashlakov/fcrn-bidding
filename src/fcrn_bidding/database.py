import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from __main__ import logger_name
from influxdb import DataFrameClient

from utils import convert_to_strf, convert_to_utc, get_start_end_time

log = logging.getLogger(logger_name)


class DBClient(object):
    """A wrapper around influx database
    Reading database query types:
        query_type="last_day"
        query_type="last_week"
        query_type="last_four_weeks"
        query_type="last_year"
        query_type="last_five_years"
        query_type="all_history"
        ....
    """

    def __init__(self, config):
        self.client = DataFrameClient(
            host=config["host"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            database=config["database"],
        )
        self.measurements = config["measurements"]
        self.glob_measurement = config["glob_measurement"]

    def read_db(
        self,
        query_type="last_day",
        field="price",
        measurement="nordicpower",
        tags={"Type": "FCR-N"},
        freq="1h",
        agg="max",
        return_local_time=False,
        offset=None,
        scale=1.0,
        fill="linear",
    ) -> pd.DataFrame:
        """
        Fetches data from influx database
        Args:
            query_type: str
        Returns:
            >>> dataframe[field]: pd.DataFrame
        """
        start_time, end_time = get_start_end_time(query_type, offset=offset)

        log.info(
            f"Fetching {field} with {list(tags.values())} tags for period from {start_time} to {end_time} (UTC time)"
        )
        # start_time = convert_to_strf(convert_to_utc(start_time))
        # end_time = convert_to_strf(convert_to_utc(end_time))
        start_time = convert_to_strf(start_time)
        end_time = convert_to_strf(end_time)

        select_clause = self.make_query(
            field_name=field,
            measurement_name=measurement,
            tags=tags,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            agg=agg,
            scale=scale,
            fill=fill,
        )
        res = self.client.query(select_clause)

        df_list, df_cols = [], []
        for key, column in res.items():
            df_list.append(res[key])
            df_cols.append(key[1][0][0])

        if df_list:
            # List is not empty
            data = pd.concat(df_list, axis=1).astype(float)  # .fillna(method="bfill")
            data.columns = [field]
        else:
            data = pd.DataFrame(columns=[field])
            return data

        if return_local_time == True:
            # convert utc time to local time
            data.index = data.index.tz_convert("Europe/Helsinki").tz_localize(None)
        else:
            data.index = data.index.tz_localize(None)
        log.debug(f"Retrieved data: {data}")
        return data

    def write_db(
        self,
        df,
        measurement_name="nordicpower",
        field_name="price",
        tags={"Type": "FCR-N"},
    ):
        """Writes to database
        Args:
        Returns:
        """
        log.info(
            f"Writing {field_name} field to {measurement_name} data with {tags} tags"
        )
        for column_name in list(df.columns.astype(str)):
            log.debug(f"Writing column: {column_name}")
            self.client.write_points(
                df[[column_name]].rename(columns={column_name: field_name}),
                measurement=measurement_name,
                field_columns=[field_name],
                tags=tags,
                protocol="line",
                batch_size=10000,
            )
        return

    def make_query(
        self,
        field_name="price",
        measurement_name="nordicpower",
        tags={"Type": "FCR-N"},
        start_time="now()",
        end_time="now()",
        freq="1h",
        agg="max",
        scale=1,
        fill="linear",
    ) -> str:
        """
        A wrapper function to generate string clauses for InfluxDBClient
        """
        # make condition string on tags
        tag = ""
        tag_name = ""
        for k, v in tags.items():
            tag += "\"{}\"='{}' AND ".format(k, v)
            tag_name += ', "{}"'.format(k)
        # make full request string
        select_clause = (
            'SELECT {}("{}")*{} FROM "{}" '
            "WHERE {} time >= '{}' AND time < '{}' "
            "GROUP BY time({}){} fill({})"
        ).format(
            agg,
            field_name,
            scale,
            measurement_name,
            tag,
            start_time,
            end_time,
            freq,
            tag_name,
            fill,
        )
        log.info(f"Requesting query: {select_clause}")
        return select_clause

    def flush_db(self, measurement_name="nordicpower", tag="FCR-N"):
        """WARNING! 'DROP' Does NOT work properly with tag"""
        log.warning(
            f"Flushing database for {measurement_name} measurement with tag {tag}"
        )
        # self.client.query('DROP SERIES FROM \"{}\" WHERE "tag" = \'{}\''.format(measurement_name, tag))
        self.client.query(f"DELETE FROM \"{measurement_name}\" WHERE Type='{tag}'")
        return
