from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
from utils import convert_to_utc, convert_to_strf, get_start_end_time
from utils import round_time, convert_kW_to_MW, convert_MW_to_kW
STATE_STANDBY = 0.
STATE_CHARGING = 1.
STATE_DISCHARGING = 2.
STATE_FCR_N = 3.

import logging
from __main__ import logger_name
log = logging.getLogger(logger_name)


class BESS_model:
    """
    Generic model class of BESS for FCRN simulation
    """

    def __init__(self, config, io):
        self.type = config["type"]
        self.name = config["name"]
        self.time_step = config["time_step"]
        self.efficiency = config["properties"]["rt_eff_perc"]*0.01
        self.up_bess_limit = config["properties"]["up_bess_limit"]
        self.low_bess_limit = config["properties"]["low_bess_limit"]
        self.max_energy_kwh = config["properties"]["max_energy_kwh"]
        self.max_power_kw = config["properties"]["max_power_kw"]
        self.init_soc_perc = config["properties"]["init_soc_perc"]
        self.properties = config["properties"]
        self.forecasts = config["forecasts"]
        self.measurements = config["measurements"]
        self.database = config["database"]
        self.io = io
        self.init_fcrn_droop_curve(config["FCRN"])

    def get_static_parameters(self):
        """
        Returns battery static parameters
        """
        return self.properties

    def init_fcrn_droop_curve(self, config):
        """
        Initialize FCR-N droop curve parameters
        """
        self.f_low_limit = config["f_low_limit"]
        self.dead_band = config["f_dead_band"]
        self.f_zero = config["f_zero"]
        self.f_high_limit = config["f_up_limit"]
        self.slope_coeff = 1 / (self.f_high_limit - (self.f_zero + self.dead_band))
        return

    def simulate_bess_fcrn_operation(self, simulation=False):
        """
        Collects frequency measurements to simulate battery FCRN response
        """
        df_dict = self.io.fetch({"frequency":["frequency"]}, request_period="last_15_minutes")
        df_dict["frequency frequency"].fillna(method="bfill", inplace=True)
        if simulation==True:
            df = self.activate(df_dict["frequency frequency"], simulation=True, max_power_kw=self.max_power_kw)
            self.io.write({"BESS-power":df[["power"]].rename(columns={"power":"power_activated"})},
                          extra_tags={"resource":self.name})
        else:
            df = self.activate(df_dict["frequency frequency"], simulation=False, max_power_kw=self.max_power_kw)
            self.io.write({"BESS-power":df[["power"]].rename(columns={"power":"power_required"})},
                          extra_tags={"resource":self.name})
        return

    def simulate_bess_operation(self, start_time, power, delta_t=1.0, simulation=False):
        """
        Simulates charge / discharge of battery storage on a second level
        """
        end_time = start_time + timedelta(seconds=int(delta_t))
        datetimes = pd.date_range(start=start_time, end=end_time, freq="1S", closed="left")
        df = pd.DataFrame(index=datetimes,
                          data=np.repeat(power,int(delta_t)),
                          columns=["p_ref"])
        df[["power", "energy"]] = df[["p_ref"]].apply(lambda x: self.get_power_energy(x["p_ref"],
                                                                                      simulation=simulation),
                                                                                      axis=1,
                                                                                      result_type="expand")
        if simulation==True:
            self.io.write({"BESS-power":df[["power"]].rename(columns={"power":"power_activated"})},
                          extra_tags={"resource":self.name})
        else:
            self.io.write({"BESS-power":df[["power"]].rename(columns={"power":"power_required"})},
                          extra_tags={"resource":self.name})
        return

    def activate(self, df, simulation=False, max_power_kw=1):
        """
        Simulates FCRN BESS control
        """
        df.rename(columns={df.columns[0]:"frequency"}, inplace=True)
        df = df.asfreq("1S", method="bfill")
        df["frequency"].replace("N/A", np.nan, inplace=True)
        df["frequency"].replace("INVA", np.nan, inplace=True)
        df["frequency"].fillna(method="ffill", inplace=True)
        df["frequency"] = df["frequency"].astype(float)
        df[["energy", "power"]] = df[["frequency"]].apply(lambda x: self.follow_frequency(x["frequency"],
                                                                                          simulation=simulation,
                                                                                          max_power_kw=max_power_kw),
                                                                                          axis=1,
                                                                                          result_type="expand")
        return df[["energy", "power"]]

    def fcrn_transform(self, df, delta_time="15T", simulation=False):
        """
        Splits the simulated FCRN parameters into variables
        """
        df = self.activate(df, simulation=simulation)
        df["up_energy"] = df["energy"]
        df["up_energy"][df["up_energy"]>=0] = 0.0
        df["down_energy"] = df["energy"]
        df["down_energy"][df["down_energy"]<=0] = 0.0
        df2 = df["up_energy"].resample(delta_time).sum()
        df3 = df["down_energy"].resample(delta_time).sum()
        df4 = df[["power"]].rename(columns={"power":"up_power"}).resample(delta_time).max()
        df4["up_power"][df4["up_power"]<=0] = 0.0
        df5 = df[["power"]].rename(columns={"power":"down_power"}).resample(delta_time).min()
        df5["down_power"][df5["down_power"]>=0] = 0.0
        df_transf = pd.concat([df2, df3, df4, df5], axis=1).fillna(method="ffill")
        return df_transf

    def follow_frequency(self, f, simulation=False, max_power_kw=1):
        """
        Maps frequency to power using droop curve parameters
        """
        if (f >= (self.f_zero - self.dead_band)) and (f <= (self.f_zero + self.dead_band)):
            power = 0.0
        elif (f > (self.f_zero + self.dead_band)):
            if f > self.f_high_limit:
                power = -max_power_kw
            else:
                power = -(f - (self.f_zero + self.dead_band)) * (self.slope_coeff*max_power_kw)
        elif (f < (self.f_zero - self.dead_band)):
            if f < self.f_low_limit:
                power = max_power_kw
            else:
                power = max_power_kw - ((f - self.f_low_limit) * (self.slope_coeff*max_power_kw))
        delta_power, delta_energy = self.get_power_energy(power, simulation=simulation)
        return round(delta_energy, 9), round(delta_power, 3)

    def get_power_energy(self, power, delta_t=1.0, simulation=False):
        """
        Returns energy change as a function of input power
        """
        power_ch, power_dc = 0.0, 0.0
        if power < 0.0:
            power_ch = power
        elif power >= 0.0:
            power_dc = power
        # convert energy to kW intervals with round trip effitiency
        delta_energy = -(power_ch * self.efficiency + power_dc / self.efficiency) * (delta_t / 3600.0)
        delta_power = power
        if simulation:
            delta_power, delta_energy = self.validate_energy_state_transition(delta_energy, delta_power, delta_t=delta_t)
        return delta_power, delta_energy

    def validate_energy_state_transition(self, delta_energy, delta_power, delta_t=1.0):
        """
        Prevents violaton of the low and high battery limits
        """
        if (self.max_energy_kwh * self.up_bess_limit) <= (self.energy_capacity + delta_energy):
            log.error(f"BESS command violates UP energy limit. required delta e: {delta_energy}, self_energy={self.energy_capacity}")
            delta_energy = (self.max_energy_kwh * self.up_bess_limit) - self.energy_capacity
            log.error(f"allowed delta e: {delta_energy}, self_energy={self.energy_capacity}")
            if delta_energy > 0.0:
                power_ch = (-delta_energy * 3600 / delta_t)/self.efficiency # 1 sec intervals
                delta_power = power_ch
            else:
                delta_energy = 0.0
                delta_power = 0.0
        elif (self.energy_capacity + delta_energy) <= (self.max_energy_kwh * self.low_bess_limit):
            log.error(f"BESS command violates LOW energy limit: required delta e: {delta_energy}, self_energy={self.energy_capacity}")
            delta_energy = (self.max_energy_kwh * self.low_bess_limit) - self.energy_capacity
            log.error(f"allowed delta e: {delta_energy}, self_energy={self.energy_capacity}")
            if delta_energy < 0.0:
                power_dc = (-delta_energy * 3600 / delta_t)*self.efficiency # 1 sec intervals
                delta_power = power_dc
            else:
                delta_energy = 0.0
                delta_power = 0.0
        self.energy_capacity += delta_energy
        return delta_power, delta_energy
