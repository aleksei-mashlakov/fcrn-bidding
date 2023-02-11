import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
import logging

from __main__ import logger_name

from battery_storage import BESS_model

log = logging.getLogger(logger_name)


class BESS(BESS_model):
    """
    Generic model class of BESS for FCRN control
    """

    def __init__(self, config, io):
        super().__init__(config, io)
        self.on_the_restart_check_up()

    def on_the_restart_check_up(self):
        """
        Restores the last saved SOC and schedule for today
        """
        self.fetch_SOC()
        self.fetch_schedule()
        return

    def fetch_SOC(self):
        """
        Fetches the schedule for today
        """
        energy_state = "BESS-energy energy_state"
        energy = self.io.fetch(
            {"BESS-energy": ["energy_state"]},
            request_period="last_four_weeks",
            extra_tags={"resource": self.name},
        )
        if energy[energy_state].empty:
            log.warning(f"{self.name} energy_state dataframe is empty")
            self.energy_capacity = (self.max_energy_kwh * self.init_soc_perc) / 100.0
        else:
            log.debug(f"{self.name} energy {energy[energy_state]}")
            self.energy_capacity = energy[energy_state].dropna().iloc[-1, :].values[0]
        log.info(f"{self.name} energy_capacity: {self.energy_capacity}")
        self.update_soc()
        log.info(f"{self.name} fetch_SOC, SOC: {self.SOC}")
        return

    def update_soc(self):
        """
        Updates BESS SOC
        """
        self.SOC = (self.energy_capacity / self.max_energy_kwh) * 100.0
        log.info(f"{self.name} update_soc, SOC: {self.SOC}")
        return

    def fetch_schedule(self):
        """
        Fetches the last saved SOC
        """
        df_dict = self.io.fetch(
            {"BESS-schedule": ["power_schedule", "state", "ref_energy_state"]},
            request_period="now_till_next_day",
            extra_tags={"resource": self.name},
        )
        df = pd.concat(list(df_dict.values()), axis=1)
        if df.empty:
            log.warning(f"{self.name} schedule dataframe is empty. Using idle values")
            end_time = datetime.utcnow().replace(
                hour=21, minute=45, second=0, microsecond=0
            )
            if not datetime.utcnow() < end_time:
                end_time += timedelta(days=1)
            start_time = round_time(datetime.utcnow())
            if start_time >= datetime.utcnow():
                start_time -= timedelta(minutes=15)
            datetimes = pd.date_range(
                start=start_time, end=end_time, freq=f"{self.time_step}T"
            ).round(f"{self.time_step}min")
            log.debug(f"{self.name} schedule datetimes: {datetimes}")
            df = pd.DataFrame(
                index=datetimes, columns=["power_schedule", "state", "ref_energy_state"]
            )
            df.loc[:, "power_schedule"] = 0.0
            df.loc[:, "state"] = 0.0
            df.loc[:, "ref_energy_state"] = 0.0
            self.io.write({"BESS-schedule": df}, extra_tags={"resource": self.name})
            self.current_state = STATE_STANDBY
            log.info(f"{self.name} bess_state_init, new_state={STATE_STANDBY}")
        else:
            df.fillna(method="ffill", inplace=True)
            str_time = round_time(datetime.utcnow())
            if str_time >= datetime.utcnow():
                str_time -= timedelta(minutes=15)
            str_time = str_time.strftime("%Y-%m-%d %H:%M:%S")
            self.current_state = self.read_inputs(time_slot=str_time)["state"][str_time]
            log.info(f"{self.name} state: {self.current_state}, time_slot: {str_time}")
        log.info(f"{self.name} restored schedule and reference state: {df}")
        return

    def update_schedule_data(self, df):
        """
        Updates battery FCRN and charge/discharge schedule
        """
        df["state"] = (df["capacity"] / convert_kW_to_MW(self.max_power_kw)).map(
            {1: STATE_FCR_N, 0: STATE_STANDBY}
        )
        df.loc[df["power_schedule"] < -1e-12, "state"] = STATE_CHARGING
        df.loc[df["power_schedule"] > 1e-12, "state"] = STATE_DISCHARGING
        log.info(f"{self.name} update_schedule_data: {df['state']}")
        return df

    def read_inputs(self, request_period=f"last_15_minutes", time_slot=None):
        """
        Read schedule power and state
        """
        df_dict = self.io.fetch(
            {"BESS-schedule": ["power_schedule", "state"]},
            request_period=request_period,
            extra_tags={"resource": self.name},
        )
        df = pd.concat(list(df_dict.values()), axis=1).dropna()
        if df.empty:
            log.warning(f"{self.name} schedule dataframe is empty, staying idle")
            df = pd.DataFrame(
                index=[time_slot],
                columns=["power_schedule", "state"],
                data=[[0.0, 0.0]],
            )
        else:
            log.info(f"{self.name} schedule: {df}, time_slot: {time_slot}")
        return df

    def update_state(self):
        """
        First, simulates the BESS operation for previous time_slot
        Then, updates the energy state and changes the schedule state
        """
        time_slot = datetime.utcnow().replace(second=0, microsecond=0) + timedelta(
            minutes=2
        )
        while datetime.utcnow() <= time_slot:
            time.sleep(1)
        str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")
        old_state = self.current_state
        df_inputs = self.read_inputs(
            request_period=f"next_{self.time_step}_minutes", time_slot=str_time
        )
        log.info(f"{self.name} update_state, time_slot={str_time}")
        if time_slot in df_inputs.index.to_list():
            new_state = df_inputs["state"][str_time]
            log.info(
                f"{self.name} bess_state_update_slot: new_state={new_state}, old_state={old_state}"
            )
        else:
            new_state = STATE_STANDBY
            log.error(
                f"{self.name} time_slot {str_time} is absent from schedule, staying idle."
            )
        self.do_the_job(time_slot)
        self.send_energy_state(time_slot)
        # self.verify_the_job()
        self.current_state = new_state
        return

    def send_energy_state(self, time_slot):
        """
        Saving battery energy state
        """
        self.update_soc()
        df_energy = pd.DataFrame(
            index=[time_slot], data=[self.energy_capacity], columns=["energy_state"]
        )
        log.info(f"{self.name} energy_state: {df_energy}")
        self.io.write({"BESS-energy": df_energy}, extra_tags={"resource": self.name})
        return

    def keep_alive(self):
        """
        Models maintainance of real BESS API control
        """
        log.info(f"{self.name} keep_alive, SOC: {self.SOC}")
        return

    def get_future_state(self):
        """
        Calculates the state of BESS for the next bidding period based on the accepted schedule
        """
        df_dict = self.io.fetch(
            {"BESS-schedule": ["ref_energy_state"]},
            request_period="now_till_next_day",
            extra_tags={"resource": self.name},
        )
        df = df_dict["BESS-schedule ref_energy_state"]
        log.debug(f"{self.name} ref_energy_state: {df}")
        future_state = self.energy_capacity + np.sum(
            df["ref_energy_state"].fillna(0.0).values
        )
        log.info(f"{self.name} get_future_state: {future_state}")
        return future_state

    def do_the_job(self, time_slot):
        """
        Shifts the state of BESS operation
        """
        time_slot -= timedelta(minutes=15)
        str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")

        if self.current_state == STATE_STANDBY:
            log.info(f"{self.name} bess_state_idle, time_slot: {str_time}")

        elif self.current_state == STATE_CHARGING:
            log.info(f"{self.name} bess_state_charging, time_slot: {str_time}")

        elif self.current_state == STATE_DISCHARGING:
            log.info(f"{self.name} bess_state_discharging, time_slot: {str_time}")

        elif self.current_state == STATE_FCR_N:
            log.info(f"{self.name} bess_state_fcrn, time_slot: {str_time}")
        return

    def verify_the_job(self):
        """
        Verifies the activation of BESS
        """
        time_slot = datetime.utcnow().replace(second=0, microsecond=0)
        time_slot -= timedelta(minutes=15)
        str_time = time_slot.strftime("%Y-%m-%d %H:%M:%S")
        inputs = self.read_inputs(time_slot=str_time)
        power = self.read_inputs(time_slot=str_time)["power_schedule"][str_time]
        state = self.read_inputs(time_slot=str_time)["state"][str_time]
        if state in [STATE_STANDBY, STATE_CHARGING, STATE_DISCHARGING]:
            # simulate previous 15 minutes using fake and reference battery
            self.simulate_bess_operation(
                time_slot, convert_MW_to_kW(power), delta_t=900.0, simulation=True
            )
            self.simulate_bess_operation(
                time_slot, convert_MW_to_kW(power), delta_t=900.0, simulation=False
            )
        else:
            self.simulate_bess_fcrn_operation(simulation=True)
            self.simulate_bess_fcrn_operation(simulation=False)
        log.info(f"{self.name} verify_the_job, finished")
        return
