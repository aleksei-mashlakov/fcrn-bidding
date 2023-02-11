import csv
import logging
import random
import socket
import time
from datetime import datetime, timedelta

import mysql.connector as mariadb
import numpy as np
import pandas as pd
import requests
import schedule
import urllib3

from battery_control import BESS
from utils import (
    convert_kW_to_MW,
    convert_MW_to_kW,
    convert_to_strf,
    convert_to_utc,
    get_start_end_time,
    round_time,
)

STATE_STANDBY = 0.0
STATE_CHARGING = 1.0
STATE_DISCHARGING = 2.0
STATE_FCR_N = 3.0

from __main__ import logger_name

log = logging.getLogger(logger_name)


class LUT_BESS(BESS):
    """
    The model of LUT Green Campus BESS for FCRN simulation with IEC104 API control
    """

    def __init__(self, config, io):
        super().__init__(config, io)
        self.enable_api_control()
        self.update_bess_parameters()

    def update_bess_parameters(self):
        """
        Set BESS SOC to nominal levels
        """
        requests.get(f"`http://localhost:5080/set/soc_limiter/min/0")
        requests.get(f"`http://localhost:5080/set/soc_limiter/max/100")
        return

    def fetch_schedule(self):
        """
        Fetches the saved schedule
        """
        df_dict = self.io.fetch(
            {"BESS-schedule": ["power_schedule", "state", "ref_energy_state"]},
            request_period="now_till_next_day",
            extra_tags={"resource": self.name},
        )
        df = pd.concat(list(df_dict.values()), axis=1)
        log.debug(f"{self.name} fetch_schedule: {df}")
        if df.empty:
            log.warning(f"{self.name} fetch_schedule is empty. Using idle values")
            end_time = datetime.utcnow().replace(
                hour=21, minute=45, second=0, microsecond=0
            )
            if not datetime.utcnow() < end_time:
                end_time += timedelta(days=1)
            time_slot = round_time(datetime.utcnow())
            if time_slot >= datetime.utcnow():
                time_slot -= timedelta(minutes=15)
            datetimes = pd.date_range(
                start=time_slot, end=end_time, freq=f"{self.time_step}T"
            ).round(f"{self.time_step}min")
            log.debug(f"{self.name} schedule_datetimes: {datetimes}")
            df = pd.DataFrame(
                index=datetimes, columns=["power_schedule", "state", "ref_energy_state"]
            )
            df.loc[:, "power_schedule"] = 0.0
            df.loc[:, "state"] = 0.0
            df.loc[:, "ref_energy_state"] = 0.0
            self.current_state = STATE_STANDBY
            power = 0.0
            self.io.write({"BESS-schedule": df}, extra_tags={"resource": self.name})
        else:
            df.fillna(method="ffill", inplace=True)
            time_slot = round_time(datetime.utcnow())
            if time_slot >= datetime.utcnow():
                time_slot -= timedelta(minutes=15)
            str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")
            self.current_state = self.read_inputs(time_slot=str_time)["state"][str_time]
            power = self.read_inputs(time_slot=str_time)["power_schedule"][str_time]
            log.info(f"{self.name} state={self.current_state}, time_slot={str_time}")
        log.info(f"{self.name} fetch_schedule {df}")
        self.do_the_job(time_slot, power)
        return

    def get_future_state(self):
        """
        Calculates the state of BESS for the next bidding period based on the accepted schedule
        """
        self.fetch_SOC()
        df_dict = self.io.fetch(
            {"BESS-schedule": ["ref_energy_state"]},
            request_period="now_till_next_day",
            extra_tags={"resource": self.name},
        )
        df = df_dict["BESS-schedule ref_energy_state"]
        log.debug(f"{self.name} ref_energy_state: {df}")
        if df.empty:
            future_state = self.energy_capacity
        else:
            future_state = self.energy_capacity + np.sum(
                df["ref_energy_state"].fillna(0.0).values
            )
        log.info(f"{self.name} energy_state: {future_state} MW")
        return future_state

    def update_state(self):
        """
        First, changes the schedule state
        Then, sends command to LUT_BESS and updates the energy state
        """
        time = datetime.utcnow().replace(second=0, microsecond=0) + timedelta(minutes=2)
        str_time = time.strftime("%Y-%m-%d %H:%M:%S")

        df_inputs = self.read_inputs(
            request_period=f"next_{self.time_step}_minutes", time_slot=str_time
        )
        old_state = self.current_state
        log.info(f"{self.name} update_state, time_slot={str_time}")
        if time in df_inputs.index.to_list():
            log.info(
                f"{self.name} bess_state_update_slot: \
                      new_state={df_inputs['state'][str_time]}, old_state={old_state}"
            )
            self.current_state = df_inputs["state"][str_time]
        else:
            log.error(
                f"{self.name} time_slot {str_time} is absent from schedule, staying idle."
            )
            self.current_state = STATE_STANDBY

        power = df_inputs["power_schedule"][str_time]
        self.do_the_job(time, power)
        return

    def do_the_job(self, time_slot, power):
        """
        Shifts the state of BESS operation
        """
        self.keep_alive()
        str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")

        if self.current_state == STATE_STANDBY:
            log.info(f"{self.name} state=bess_state_idle, time slot={str_time}")
            requests.get(
                f"`http://localhost:5080/set/p_ref/{int(convert_MW_to_kW(abs(power)))}"
            )
            requests.get("`http://localhost:5080/set/api_control_task/idle")
            self.apollo()

        elif self.current_state == STATE_CHARGING:
            log.info(f"{self.name} state=bess_state_charging, time slot={str_time}")
            requests.get(
                f"`http://localhost:5080/set/p_ref/{int(convert_MW_to_kW(abs(power)))}"
            )
            requests.get("`http://localhost:5080/set/api_control_task/charge")

        elif self.current_state == STATE_DISCHARGING:
            log.info(f"{self.name} state=bess_state_discharging, time slot={str_time}")
            requests.get(
                f"`http://localhost:5080/set/p_ref/{int(convert_MW_to_kW(abs(power)))}"
            )
            requests.get("`http://localhost:5080/set/api_control_task/discharge")

        elif self.current_state == STATE_FCR_N:
            log.info(f"{self.name} state=bess_state_fcrn, time slot={str_time}")
            requests.get(
                f"`http://localhost:5080/set/p_ref/{int(0.83*self.max_power_kw)}"
            )
            requests.get("`http://localhost:5080/set/api_control_task/FCRN")

        p_ref = requests.get("`http://localhost:5080/get/p_ref").json()["LVDC_gc"][
            "p_ref"
        ]
        task = requests.get("`http://localhost:5080/get/api_control_task").json()[
            "api_control_task"
        ]
        log.info(f"{self.name} task={task}, power={p_ref} kW")
        return

    def verify_the_job(self):
        """
        Verifies the activation of BESS for schedules state
        """
        time_slot = datetime.utcnow().replace(second=0, microsecond=0)
        self.send_energy_state(time_slot)
        # simulate previous 15 minutes
        time_slot -= timedelta(minutes=15)
        str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")
        df_inputs = self.read_inputs(
            request_period=f"last_{self.time_step}_minutes", time_slot=str_time
        )
        old_state = df_inputs["state"][str_time]
        # get and save the required BESS power measurements
        if old_state in [STATE_STANDBY, STATE_CHARGING, STATE_DISCHARGING]:
            power = df_inputs["power_schedule"][str_time]
            self.simulate_bess_operation(
                time_slot, convert_MW_to_kW(power), delta_t=900.0, simulation=False
            )
        else:
            self.simulate_bess_fcrn_operation(simulation=False)
        # get and save the actual BESS power measurements
        self.get_bess_power(time_slot)
        return

    def send_energy_state(self, time):
        """
        Saving battery energy state
        """
        self.fetch_SOC()
        df_energy = pd.DataFrame(
            index=[time], data=[self.energy_capacity], columns=["energy_state"]
        )
        log.info(f"{self.name} energy_state: {df_energy}")
        self.io.write({"BESS-energy": df_energy}, extra_tags={"resource": self.name})
        return

    def get_bess_power(self, start_time):
        """
        Fetches the power measurements from BESS
        """
        mariadb_connection = mariadb.connect(
            host=self.database["host"],
            port=self.database["port"],
            user=self.database["user"],
            password=self.database["password"],
            database=self.database["database"],
        )
        cursor = mariadb_connection.cursor()
        # retrieving information
        end_time = start_time + timedelta(seconds=900)
        start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute(
                f'SELECT P_BESS_AVG, TIMESTAMP FROM BESS_MC WHERE TIMESTAMP BETWEEN "{start_time}" AND "{end_time}"'
            )
        except mariadb.Error as e:
            log.error(f"{self.name} MAriaDB Error: {e}")
        result = cursor.fetchall()
        data = []
        for row in result:
            data.append(row)
        cursor.close()
        df = pd.DataFrame(data, columns=["power_activated", "Time"]).set_index("Time")
        # to convert the measurements from net load side to bess side
        df *= -1
        log.info(f"{self.name} activated power {df} (raw)")
        df = df.resample("1S").mean()
        log.info(f"{self.name} activated power {df} (resampled)")
        self.io.write({"BESS-power": df}, extra_tags={"resource": self.name})
        mariadb_connection.close()
        return

    def fetch_SOC(self):
        """
        Reads BESS SOC state
        """
        r = requests.get("`http://localhost:5080/get/soc")
        soc = r.json()["LVDC_gc"]["soc"]
        if soc != "None":
            self.SOC = r.json()["LVDC_gc"]["soc"]
            log.info(f"{self.name} bess_soc_state: soc={self.SOC}")
            self.energy_capacity = (self.max_energy_kwh * self.SOC) / 100.0
        else:
            log.warning(f"bess_soc_state is unidentified, soc={soc}")
        return

    def keep_alive(self):
        """
        Maintains API control over BESS
        """
        requests.get("`http://localhost:5080/keep_alive")
        self.fetch_SOC()
        log.info(f"{self.name} keep_alive, SOC value: {self.SOC}")

    def deactivate(self):
        """
        Sets idle state and terminates BESS active API control
        """
        self.end()
        self.disable_api_control()

    def end(self):
        """
        Shifts BESS to idle state
        """
        log.info(
            f"{self.name} bess_state_update: \
                   new_state={'STATE_STANDBY'}, \
                   old_state={self.current_state}"
        )
        self.current_state = STATE_STANDBY
        requests.get("`http://localhost:5080/set/api_control_task/idle")

    def disable_api_control(self):
        """
        Disables API control over BESS
        """
        requests.get("`http://localhost:5080/disable/api_control")

    def enable_api_control(self):
        """
        Runs LUT BESS according to the Flask API
        """
        requests.get("`http://localhost:5080/enable/api_control")
        return

    def apollo(self):
        """
        Maintains BESS state-of-charge within the device limits
        """
        if self.SOC >= int(self.up_bess_limit * 100):
            log.info(
                "bess_state_update_high_soc: soc={}, new_state={}, old_state={}".format(
                    self.SOC, "STATE_DISCHARGING", self.current_state
                )
            )
            self.current_state = STATE_DISCHARGING
            requests.get(f"`http://localhost:5080/set/p_ref/5")
            requests.get("`http://localhost:5080/set/api_control_task/discharge")
            return
        elif self.SOC < int(self.low_bess_limit * 100):
            log.info(
                "bess_state_update_low_soc: soc={}, new_state={}, old_state={}".format(
                    self.SOC, "STATE_CHARGING", self.current_state
                )
            )
            self.current_state = STATE_CHARGING
            requests.get(f"`http://localhost:5080/set/p_ref/5")
            requests.get("`http://localhost:5080/set/api_control_task/charge")
            return
        else:
            return
