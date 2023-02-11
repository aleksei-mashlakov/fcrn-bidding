"""
This module contains project-specific io functionality.
"""
import logging

from __main__ import logger_name

from database import DBClient
from fingridapi import FingridAPI

log = logging.getLogger(logger_name)


class IO_handler(object):
    """Handles influx databases and Fingrid API"""

    def __init__(self, config):
        """
        Initializes the IO handler
        Args:
            config: configuration file for the IO objects
        """
        self.fingrid_api = FingridAPI(config["api"]["fingrid"])
        self.database_clients = {k: DBClient(v) for k, v in config["databases"].items()}
        self.measurement_to_clients = {
            m_k: cl_k
            for cl_k, cl_v in self.database_clients.items()
            for m_k, m_v in cl_v.measurements.items()
        }
        log.debug(f"measurement_to_clients: {self.measurement_to_clients}")

    def update_fingrid_api(self, meas_name="FCRN", request_period="last_day"):
        """
        Updates the database points with new Fingrid measurements
        """
        log.info(f"Fetching the {meas_name} prices")
        db_client = self.database_clients[self.measurement_to_clients[meas_name]]
        for field in db_client.measurements[meas_name]["fields"]:
            request_type = meas_name + " " + field
            log.info(f"Updating the database with {request_type} data")
            df = self.fingrid_api.request_fingrid_data(
                type=request_type,
                request_period=request_period,
                return_local_time=False,
            )
            log.debug(f"Retrieved Fingrid price data: {df}")
            db_client.write_db(
                df,
                measurement_name=db_client.glob_measurement,
                field_name=field,
                tags=db_client.measurements[meas_name]["tags"],
            )
        return

    def fetch(
        self,
        request_data,
        request_period="all_history",
        return_local=False,
        offset=None,
        scale=1,
        extra_tags={},
    ) -> dict:
        """
        Fetches the data from database clients
        Args:
            request_data - dict of {measurement: [field1, field2]}
            request_period - time period of the measurements
        Returns:
            >>> dictionary of dataframes {measurement+' '+field:df}
        """
        df_dict = dict()
        log.info(f"Fetching {request_data} data")
        for meas_name, fields in request_data.items():
            for field in fields:
                db_client = self.database_clients[
                    self.measurement_to_clients[meas_name]
                ]
                request_type = meas_name + " " + field
                log.debug(f"Fetching the database with {request_type} data")
                df_dict[request_type] = db_client.read_db(
                    query_type=request_period,
                    field=field,
                    measurement=db_client.glob_measurement,
                    tags={**db_client.measurements[meas_name]["tags"], **extra_tags},
                    freq=db_client.measurements[meas_name]["freq"],
                    agg=db_client.measurements[meas_name]["agg"],
                    return_local_time=return_local,
                    offset=offset,
                    scale=scale,
                    fill=db_client.measurements[meas_name]["fill"],
                )
        log.debug(f"Obtained data {df_dict}")
        return df_dict

    def write(self, dfs_dict, extra_tags={}):
        """
        Writes the dataframe data to databases
        Args:
            dfs_dict - dictionary of dataframes to write {'measurement':df[fields]}
        """
        for meas_name, df in dfs_dict.items():
            db_client = self.database_clients[self.measurement_to_clients[meas_name]]
            if df.columns.isin(db_client.measurements[meas_name]["fields"]).all():
                for column in list(df.columns):
                    log.info(
                        f"Writing the {column} field to the {db_client.glob_measurement} measurement"
                    )
                    db_client.write_db(
                        df[[column]],
                        measurement_name=db_client.glob_measurement,
                        field_name=column,
                        tags={
                            **db_client.measurements[meas_name]["tags"],
                            **extra_tags,
                        },
                    )
        return
