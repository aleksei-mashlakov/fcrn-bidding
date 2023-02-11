import logging
from datetime import timedelta

import pandas as pd
from __main__ import logger_name

from bidding import Bid
from io_handler import IO_handler
from MsgLog import LogInit
from prediction import Predictor
from utils import reindex_dataframe, with_logging

log = logging.getLogger(logger_name)


class Proxy(object):
    """Mediator class between pipeline and other classes"""

    def __init__(self, config):
        self.config = config
        self.Bid = Bid(self.config["bidding"])
        self.IO_handler = IO_handler(self.config["io"])
        self.Predictor = Predictor(self.config["prediction"])
        self.Devices = {k: self.init_device(v) for k, v in config["devices"].items()}

    def init_device(self, config):
        if config["mode"] == "real_testing":
            from lut_bess import LUT_BESS

            bess = LUT_BESS(config, self.IO_handler)
        elif config["mode"] == "simulation":
            from battery_control import BESS

            bess = BESS(config, self.IO_handler)
        else:
            log.error(f"{config.name} is missing mode from configuration")
        return bess

    @with_logging
    def keep_alive(self):
        [device.keep_alive() for device in list(self.Devices.values())]
        return

    @with_logging
    def update_fingrid_api(self, request_period="last_day"):
        self.IO_handler.update_fingrid_api(request_period=request_period)
        return

    @with_logging
    def update_state(self):
        [device.update_state() for device in list(self.Devices.values())]
        return

    @with_logging
    def verify_the_job(self):
        [device.verify_the_job() for device in list(self.Devices.values())]
        return

    @with_logging
    def submit_bids(self):
        """
        Calculates a bid for the FCR-N market and stores it in DB
        """
        fcrn_forecasts = self.make_FCRN_forecasts()
        for device in list(self.Devices.values()):
            device_forecasts = self.make_device_forecasts(device=device)
            forecasts = {**fcrn_forecasts, **device_forecasts}
            log.info(f"{device.name} forecasts {forecasts}")
            bid = self.Bid.create(forecasts, device=device)
            df_bid = pd.DataFrame(
                index=pd.to_datetime(bid["hour"]),
                data=bid["capacity"],
                columns=["FCRN_capacity"],
            )
            log.info(f"{device.name} BID-submitted: {df_bid}")
            self.IO_handler.write(
                {"BID-submitted": df_bid}, extra_tags={"resource": device.name}
            )
            log.error(f"{device.name} BID-accepted (Fake)")
            self.IO_handler.write(
                {"BID-accepted": df_bid}, extra_tags={"resource": device.name}
            )
        return

    @with_logging
    def update_bids(self, request_period="next_day"):
        """
        Updates FCR-N bids with accepted values for the BESS control
        """
        self.update_fingrid_api(request_period=request_period)
        acc = self.get_accepted_bids(request_period=request_period)
        fcrn_forecasts = self.update_forecasts(request_period=request_period)
        for device in list(self.Devices.values()):
            forecasts = {**fcrn_forecasts, **self.get_device_forecasts(device=device)}
            log.info(f"{device.name} forecasts {forecasts}")
            df_dict = self.IO_handler.fetch(
                {"BID-submitted": ["FCRN_capacity"]},
                request_period=request_period,
                return_local=False,
                extra_tags={"resource": device.name},
            )
            df = (
                df_dict["BID-submitted FCRN_capacity"]
                .rename(columns={"FCRN_capacity": "capacity"})
                .fillna(0.0)
            )
            log.debug(f"{device.name} BID-submitted: {df}")
            df = df * acc.values
            log.info(f"{device.name} BID-accepted: {df}")
            power_energy = self.Bid.update_bids(df, forecasts, device=device)
            log.info(f"power_energy {power_energy}")
            df["power_schedule"] = power_energy[0]
            df["ref_energy_state"] = power_energy[1]
            df = device.update_schedule_data(df)  # power state ref_energy
            log.info(f"{device.name} BESS-schedule: {df}")
            self.IO_handler.write(
                {"BESS-schedule": df[["power_schedule", "state", "ref_energy_state"]]},
                extra_tags={"resource": device.name},
            )
        return

    def get_accepted_bids(self, request_period="next_day"):
        """
        Returns a binary sequence of accepted bids per time slot
        """
        dict_accept = self.IO_handler.fetch(
            {"BID-accepted": ["FCRN_capacity"]},
            request_period=request_period,
            return_local=False,
        )
        log.debug(f"Total accepted_capacity: {dict_accept}")
        dict_submit = self.IO_handler.fetch(
            {"BID-submitted": ["FCRN_capacity"]},
            request_period=request_period,
            return_local=False,
        )
        log.debug(f"Total submitted_capacity: {dict_submit}")
        acc = dict_accept["BID-accepted FCRN_capacity"].fillna(0.0)
        acc.loc[acc[acc["FCRN_capacity"] > 0.0].index, "FCRN_capacity"] = 1.0
        log.debug(f"BID-accepted {acc}")
        return acc

    def update_forecasts(self, request_period="next_day"):
        """
        Replases price forecast with actual data and produces fresh BESS energy forecast
        """
        fcrn_forecasts = self.get_FCRN_forecasts()
        log.debug(f"FCRN forecasts: {fcrn_forecasts}")
        actual_prices = self.get_fcrn_price_time_series(request_period=request_period)[
            "FCRN price"
        ]
        log.debug(f"Actual_FCRN_prices: {actual_prices}")
        fcrn_forecasts["price"] = reindex_dataframe(actual_prices).rename(
            columns={"price": "q50"}
        )
        actual_prices = self.get_wholesale_price_time_series(
            request_period=request_period
        )["wholesale price"]
        log.debug(f"Actual_WS_prices: {actual_prices}")
        fcrn_forecasts["wholesale_price"] = reindex_dataframe(actual_prices).rename(
            columns={"wholesale_price": "q50"}
        )
        log.debug(f"Actual_WS_prices: {actual_prices}")
        df_bess_up_down = self.forecast_bess_fcrn_response(
            request_period="last_two_weeks", transform_period="last_day", write=False
        )
        fcrn_forecasts["up_energy"] = df_bess_up_down["up_energy"]
        fcrn_forecasts["down_energy"] = df_bess_up_down["down_energy"]
        return fcrn_forecasts

    def get_fcrn_price_time_series(self, request_period="last_year") -> dict:
        """
        Fetches FCRN price from database
        """
        df_dict = self.IO_handler.fetch(
            {"FCRN": ["price"]}, request_period=request_period
        )
        log.debug(f"fcrn price: {df_dict}")
        df_dict["FCRN price"] = df_dict["FCRN price"].ffill().bfill()
        log.info(f"request_period={request_period}, FCR-N price: {df_dict}")
        return df_dict

    def get_wholesale_price_time_series(self, request_period="last_year") -> dict:
        """
        Fetches FCRN price from database
        """
        df_dict = self.IO_handler.fetch(
            {"wholesale": ["price"]}, request_period=request_period
        )
        log.debug(f"wholesale price: {df_dict}")
        df_dict["wholesale price"] = df_dict["wholesale price"].ffill().bfill()
        df_dict["wholesale price"] = df_dict["wholesale price"].rename(
            columns={"price": "wholesale_price"}
        )
        log.info(f"request_period={request_period}, wholesale price: {df_dict}")
        return df_dict

    def get_bess_fcrn_time_series(self, request_period="last_week") -> dict:
        """
        Fetches BESS energy FCRN response from database
        """
        df_dict = self.IO_handler.fetch(
            {
                "BESS": [
                    "up_energy",
                    "down_energy",
                    # "up_power",
                    # "down_power"
                ]
            },
            request_period=request_period,
            offset="1D",
        )
        for key, df in df_dict.items():
            df.dropna(axis=0, how="all", inplace=True)
            start = df.index[0].replace(hour=23, minute=0, second=0, microsecond=0)
            end = df.index[-1].replace(hour=22, minute=59, second=59, microsecond=0)
            if key == "BESS down_power" or key == "BESS up_energy":
                df = -df
            df_dict[key] = df[start:end]
        log.debug(
            f"get_bess_fcrn_time_series,  request_period={request_period}: {df_dict}"
        )
        return df_dict

    def get_load_time_series(
        self, measurement, request_period="last_year", offset=None, scale=1
    ) -> dict:
        """
        Fetches device load from database
        """
        if not offset == None:
            offset = "1D " + offset
        else:
            offset = "1D"
        df_dict = self.IO_handler.fetch(
            measurement, request_period=request_period, offset=offset, scale=scale
        )
        key = (
            str(list(measurement.keys())[0])
            + " "
            + str(list(measurement.values())[0][0])
        )
        df_dict[key] = df_dict[key].ffill().bfill()
        log.info(
            f"measurement={measurement}, request_period={request_period}, load={df_dict[key]}"
        )
        return df_dict

    @with_logging
    def train_models(self):
        """
        Trains the prediction models
        """
        for model_name, targets in self.Predictor.map_model_to_targets.items():
            if model_name == "deepar1":
                df_dict = self.get_fcrn_price_time_series(
                    request_period="last_five_years"
                )
            elif model_name == "deepar2":
                df_dict = self.get_bess_fcrn_time_series(request_period="last_year")
            elif model_name == "deepar3":
                df_dict = self.get_load_time_series(
                    {"lut": ["total"]}, scale=0.2, request_period="last_three_years"
                )
            elif model_name == "deepar4":
                df_dict = self.get_load_time_series(
                    {"LVDC-load": ["amr_kw"]}, offset="3Y", request_period="last_year"
                )
            elif model_name == "deepar5":
                df_dict = self.get_wholesale_price_time_series(
                    request_period="last_three_years"
                )
            else:
                log.error(f"model_name {model_name} is unknown. Revise specification")
            self.Predictor.train(df_dict)
        return

    def make_FCRN_forecasts(self) -> dict:
        """
        Produces FCRN-related forecasts
        """
        dict1 = self.forecast_fcrn_price(request_period="last_two_weeks")
        dict2 = self.forecast_wholesale_price(request_period="last_two_weeks")
        dict3 = self.forecast_bess_fcrn_response(request_period="last_two_weeks")
        return {**dict1, **dict2, **dict3}

    def make_device_forecasts(self, device) -> dict:
        """
        Produce device-specific forecasts (load and solar generation)
        """
        if device.name == "LUT_BESS":
            dict1 = self.get_solar_production_forecast(
                request_period="next_day", offset="1Y", scale=70.0
            )
            dict2 = self.forecast_load(
                device, request_period="last_two_weeks", scale=0.2
            )
        elif device.name == "LVDC_BESS":
            dict1 = self.get_solar_production_forecast(
                request_period="next_day", offset="1Y", scale=10.0
            )
            dict2 = self.forecast_load(
                device, offset="3Y", request_period="last_two_weeks"
            )
        return {**dict1, **dict2}

    def get_FCRN_forecasts(self) -> dict:
        """
        Wraps FCRN forecast methods
        """
        dict1 = self.get_fcrn_price_forecast(request_period="next_day")
        dict2 = self.get_wholesale_price_forecast(request_period="next_day")
        dict3 = self.get_bess_fcrn_forecast(request_period="next_day")
        return {**dict1, **dict2, **dict2}

    def get_device_forecasts(self, device) -> dict():
        """
        Wraps device-specific forecasts (load and solar generation)
        """
        if device.name == "LUT_BESS":
            dict1 = self.get_solar_production_forecast(
                request_period="next_day", offset="1Y", scale=70.0
            )
            dict2 = self.get_load_forecast(device, request_period="next_day")
        elif device.name == "LVDC_BESS":
            dict1 = self.get_solar_production_forecast(
                request_period="next_day", offset="1Y", scale=10.0
            )
            dict2 = self.get_load_forecast(
                device, request_period="next_day", offset="3Y"
            )
        return {**dict1, **dict2}

    def forecast_fcrn_price(self, request_period="last_year") -> dict:
        """
        Predicts the FCR-N market price and stores it in DB
        """
        df_dict = self.get_fcrn_price_time_series(request_period=request_period)
        log.info("Forecasting FCR-N price ... ")
        forecasts = self.Predictor.predict(df_dict)
        log.info(f"Price forecasts: {forecasts}")
        forecasts["price"] = reindex_dataframe(forecasts["price"])
        self.IO_handler.write({"FCRN-price-frcst": forecasts["price"]})
        return forecasts

    def forecast_wholesale_price(self, request_period="last_year") -> dict:
        """
        Predicts the wholesale market price and stores it in DB
        """
        df_dict = self.get_wholesale_price_time_series(request_period=request_period)
        log.info("Forecasting wholesale market price ... ")
        forecasts = self.Predictor.predict(df_dict)
        log.info(f"Wholesale price forecasts: {forecasts}")
        forecasts["wholesale_price"] = reindex_dataframe(forecasts["wholesale_price"])
        self.IO_handler.write({"WS-price-frcst": forecasts["wholesale_price"]})
        return forecasts

    def forecast_bess_fcrn_response(
        self,
        request_period="last_three_months",
        transform_period="day_before",
        write=True,
    ) -> dict:
        """
        Predicts the BESS response on the FCR-N market
        """
        log.info(
            f"request_period={request_period}, transform_period={transform_period}"
        )
        self.transform_frequency(request_period=transform_period)
        df_dict = self.get_bess_fcrn_time_series(request_period=request_period)
        log.debug(f"Forecasting BESS FCR-N response {df_dict.keys()}")
        forecasts = self.Predictor.predict(df_dict)
        #     log.warning(f"Forecasting with Naive model")
        #     forecasts = {}
        #     for label, df in df_dict.items():
        #         log.debug(f"label {label}, df {df}")
        #         df = reindex_dataframe(df, method=None, freq="15T", delta=f"{192*15+15}T")
        #         log.debug(f"concat reindexed datframe {df}")
        #         col_name = str(df.columns[0])
        #         frcst_df = self.Predictor.naive_weekly_seasonal(df, horizon=192)
        #         forecasts[col_name] = frcst_df.rename(columns={col_name:"q50"})[-96:]
        log.info(f"BESS FCR-N response forecasts: {forecasts}")
        for key, df in forecasts.items():
            # if key=="down_power" or key=="up_power":
            #     for column in forecasts[key].columns:
            #         index_mask = forecasts[key][column]>self.Device.max_power_kw
            #         forecasts[key].loc[index_mask, column] = self.Device.max_power_kw
            if key == "down_energy" or key == "up_energy":
                for column in forecasts[key].columns:
                    forecasts[key].loc[df[column] < 0.0, column] = 0.0
            if key == "down_power" or key == "up_energy":
                forecasts[key] = -forecasts[key]
            log.debug(f"forecasts {key}: {forecasts[key]}")
            if write:
                self.IO_handler.write({f"FCRN-{key}-frcst": forecasts[key].iloc[-96:]})
            else:
                forecasts[key] = forecasts[key].iloc[:-96]
        return forecasts

    def forecast_load(
        self, device, request_period="last_year", offset=None, scale=1.0
    ) -> dict:
        """
        Predicts the device load, stores it in DB, and sends further
        """
        df_dict = self.get_load_time_series(
            device.measurements["load"],
            request_period=request_period,
            offset=offset,
            scale=scale,
        )
        forecasts = self.Predictor.predict(df_dict)
        key = list(forecasts.keys())[0]
        if forecasts[key].index.freq == "H":
            forecasts[key] = reindex_dataframe(forecasts[key])
        log.info(f"{device.name} forecasted load: {forecasts[key]}")
        self.IO_handler.write({device.forecasts["load"]: forecasts[key]})
        return {"load": forecasts[key]}

    def transform_frequency(self, request_period="day_before"):
        """ """
        df_dict = self.IO_handler.fetch(
            {"frequency": ["frequency"]}, request_period=request_period
        )
        # df_dict["93301 frequency"].fillna(method="bfill", inplace=True)
        log.debug(f"frequency data: {df_dict}")
        meas_field = "frequency frequency"
        df_dict[meas_field].dropna(inplace=True)
        df_dict[meas_field] = reindex_dataframe(
            df_dict[meas_field], method="bfill", freq="1S", delta="1S"
        ).ffill()
        if not (request_period == "day_before" or request_period == "last_day"):
            start = (
                df_dict[meas_field]
                .index[0]
                .replace(hour=23, minute=0, second=0, microsecond=0)
            )
            end = df_dict[meas_field].index[-1].replace(
                hour=22, minute=59, second=59, microsecond=0
            ) - timedelta(days=1)
            df_dict[meas_field] = df_dict[meas_field][start:end]
        log.debug(f"processed frequency data {df_dict[meas_field]}")
        self.IO_handler.write(
            {
                "BESS": list(self.Devices.values())[0].fcrn_transform(
                    df_dict[meas_field], delta_time="15T"
                )
            }
        )
        return

    def get_solar_production_forecast(
        self, request_period="next_day", scale=84.0, offset=None
    ) -> dict:
        df_ = self.IO_handler.fetch(
            {"solar-forecast": ["kwh"]},
            request_period=request_period,
            offset=offset,
            scale=scale,
        )
        df = (
            df_["solar-forecast kwh"]
            .resample("15T")
            .ffill()
            .rename(columns={"kwh": "solar_production_kwh"})
        )
        return {"solar_production_kwh": reindex_dataframe(df)}

    def get_fcrn_price_forecast(self, request_period="next_day") -> dict:
        df_ = self.IO_handler.fetch(
            {"FCRN-price-frcst": ["q25", "q50", "q75"]}, request_period=request_period
        )
        df = df_["FCRN-price-frcst q50"].ffill().bfill()
        return {"price": df}

    def get_wholesale_price_forecast(self, request_period="next_day") -> dict:
        df_ = self.IO_handler.fetch(
            {"WS-price-frcst": ["q25", "q50", "q75"]}, request_period=request_period
        )
        df = df_["WS-price-frcst q50"].ffill().bfill()
        return {"wholesale_price": df}

    def get_bess_fcrn_forecast(self, request_period="next_day") -> dict:
        df_ = self.IO_handler.fetch(
            {
                "FCRN-up_energy-frcst": ["q25", "q50", "q75"],
                "FCRN-down_energy-frcst": ["q25", "q50", "q75"],
            },
            request_period=request_period,
        )
        df_up = df_["FCRN-up_energy-frcst q75"].ffill().bfill()
        df_down = df_["FCRN-down_energy-frcst q75"].ffill().bfill()
        return {"up_energy": df_up, "down_energy": df_down}

    def get_load_forecast(self, device, offset=None, request_period="next_day") -> dict:
        df_ = self.IO_handler.fetch(
            {device.forecasts["load"]: ["q25", "q50", "q75"]},
            offset=offset,
            request_period=request_period,
        )
        df = df_[device.forecasts["load"] + " q50"].ffill().bfill()
        return {"load": df}
