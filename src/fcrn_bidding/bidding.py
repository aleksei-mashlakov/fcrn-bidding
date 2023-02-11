"""
This module contains generic functionality to carry out a FCR-N bidding process based on CVXPY library
"""
import logging
from datetime import datetime, timedelta

import cplex
import cvxpy as cp
import numpy as np
import pandas as pd
from __main__ import logger_name

from io_handler import IO_handler
from utils import convert_kW_to_MW, convert_to_euro, round_results

log = logging.getLogger(logger_name)


class Bid(object):
    """
    Runs convex optimization for BESS participating in FCR-N market and
    forms a bid to the market
    """

    def __init__(self, config):

        self.forecasts = None
        self.T = config["period"]
        self.dt = config["dt"]

    def create(self, forecasts, market="FCRN", device=None):
        """Forms a bid for a single BESS to the FCR-N market
        A bid must contain the following information:
            * product  (FCR-N)
            * capacity (MW)
            * price of availability throughout the agreement period (€/MW,h)
            * hour of the agreement period
            * name or list of the Reserve Units
        """
        self.forecasts = forecasts
        bid = {
            "product": market,
            "capacity": self.get_capacity(device).tolist(),
            "price": self.get_forecasts(type="price", quantile="q50").tolist(),
            "hour": self.forecasts["price"].index.tolist(),
            "name": device.name,
        }
        log.info(f"{device.name} bid: {bid}")
        return bid

    def update_bids(self, df, forecasts, device):
        """Return battery power schedule"""
        self.forecasts = forecasts
        p_max = convert_kW_to_MW(device.get_static_parameters()["max_power_kw"])
        mask = df["capacity"].values / p_max
        log.info(f"{device.name} accepted_capacity {mask}")
        (q, b, c_reg_share) = self.optimize_bess(device, reoptimize=True, mask=mask)
        log.debug(
            f"{device.name} schedules: {[b.tolist(), np.diff(np.array(q.tolist()))]}"
        )
        return [b.tolist(), np.diff(np.array(q.tolist()))]

    def get_capacity(self, device):
        """Calculates bidding bess power capacity"""
        (q, b, c_reg_share) = self.optimize_bess(device)
        p_max = device.get_static_parameters()["max_power_kw"]
        bidding_capacity = c_reg_share * convert_kW_to_MW(p_max)
        return round_results(bidding_capacity, decimals=5)

    def get_solar_production_forecast(self):
        """Returns solar forecast"""
        return self.forecasts["solar_production_kwh"].values

    def get_forecasts(self, type="price", quantile="q50"):
        """A simple method to get the forecast array
        Args:
            type: the type of the forecast.
        Returns:
            A forecast
        """
        return self.forecasts[type].loc[:, quantile].values

    def optimize_bess(self, device=None, test=False, reoptimize=False, mask=None):
        """Finds a solution to a convex optimization problem
        Warning! For CVXPY operations:
            Use ``*`` for matrix-scalar and vector-scalar multiplication.
            Use ``@`` for matrix-matrix and matrix-vector multiplication.
            Use ``multiply`` for elementwise multiplication.
        """
        # Generate a random problem
        np.random.seed(0)
        ## BATTERY STORAGE
        params = device.get_static_parameters()
        log.debug(f"{device.name} parameters: {params}")
        P = convert_kW_to_MW(params["max_power_kw"])
        E = convert_kW_to_MW(params["max_energy_kwh"])
        # Constants
        C = cp.Constant(value=np.repeat(P, self.T))
        Q = cp.Constant(E)
        eff = cp.Constant(params["rt_eff_perc"] * 0.01)
        Q_min = cp.Constant(E * params["low_bess_limit"])
        Q_max = cp.Constant(E * params["up_bess_limit"])

        Q_start = cp.Constant(convert_kW_to_MW(device.get_future_state()))
        log.info(f"{device.name} get_future_state: {Q_start} MW")

        Q_end = cp.Constant(E * params["next_day_soc"])
        # Variables
        q = cp.Variable(self.T + 1, name="BESS energy")
        b = cp.Variable(self.T, name="BESS power")
        bc = cp.Variable(self.T, name="BESS charging Power")
        bd = cp.Variable(self.T, name="BESS discharging Power")
        b_ch_dis = cp.Variable(self.T, boolean=True, name="BESS CD boolean")

        ## NET LOAD
        # Variables
        s_gen = convert_kW_to_MW(
            self.get_solar_production_forecast().reshape(
                self.T,
            )
        )
        log.debug(f"{device.name} s_gen: {s_gen} MW")
        load = convert_kW_to_MW(
            self.get_forecasts(type="load", quantile="q50")[-self.T :].reshape(
                self.T,
            )
        )
        log.debug(f"{device.name} load: {load} MW")
        nl = cp.Variable(self.T, name="Net load power")
        nl_neg = cp.Variable(self.T, name="Net load negative")
        nl_pos = cp.Variable(self.T, name="Net load positive")
        nl_pos_neg = cp.Variable(self.T, boolean=True, name="Net load boolean")
        # Constansts
        nl_fuse = cp.Constant(
            np.repeat(params["grid_capacity_mw"], self.T).reshape(
                self.T,
            )
        )

        ## FREQUENCY RESPONSE
        # Variables
        if reoptimize:
            c_reg_share = cp.Constant(value=mask)
            log.info(f"{device.name} c_reg_share: {c_reg_share.value}")
        else:
            # !boolean=False if capacity can be divided for several states
            c_reg_share = cp.Variable(self.T, boolean=True, name="FCR-N power capacity")
        z = cp.Variable(self.T, name="Boolean variable for charging")
        z1 = cp.Variable(self.T, name="Boolean variable for discharging")

        # down reg - neg power - positive energy - charge battery - withdraw power to the grid
        # up reg - pos power - neg energy - discharge battery - supply power to the grid

        # C_reg_up = convert_kW_to_MW(self.get_forecasts(type="up_power", quantile="q75")[-self.T:].reshape(self.T,))
        # C_reg_down = convert_kW_to_MW(self.get_forecasts(type="down_power", quantile="q75")[-self.T:].reshape(self.T,))
        Q_reg_up = -convert_kW_to_MW(
            params["max_power_kw"]
            * self.get_forecasts(type="up_energy", quantile="q50")[-self.T :].reshape(
                self.T,
            )
        )
        Q_reg_down = convert_kW_to_MW(
            params["max_power_kw"]
            * self.get_forecasts(type="down_energy", quantile="q50")[-self.T :].reshape(
                self.T,
            )
        )
        log.debug(f"{device.name} Q_reg_up: {Q_reg_up}")
        log.debug(f"{device.name} Q_reg_down: {Q_reg_down}")

        ## Create constraints
        constraints = [  ## Model the battery storage
            q[0] == Q_start,
            q[self.T] >= Q_end,  # $q_{t+1} = q_t + c_t$, $t = 1, \ldots, self.T − 1$,
            cp.diff(q)
            == -(bc * eff + bd / eff) * self.dt
            + cp.multiply(c_reg_share, (Q_reg_down - Q_reg_up)),
            -q <= -Q_min,
            q <= Q_max,
            bc <= 0,
            -bd <= 0,
            b == bc + bd,
            ## Linearize binary and Variable product for bd <= cp.multiply(b_ch_dis, (C - cp.multiply(c_reg_share, C))),
            z >= 0,
            z <= b_ch_dis,
            cp.multiply((1 - b_ch_dis), (-C)) <= z - c_reg_share,
            # cp.multiply((1 - b_ch_dis), (-C_reg_up)) <= z - c_reg_share,
            z - c_reg_share <= 0,
            bd <= cp.multiply(b_ch_dis, C) - cp.multiply(z, C),
            ## Linearize binary and variable product for bc >= cp.multiply((b_ch_dis-1), (C - cp.multiply(c_reg_share, C))),
            z1 >= 0,
            z1 <= b_ch_dis,
            -(1 - b_ch_dis) <= z1 - c_reg_share,
            z1 - c_reg_share <= 0,
            bc
            >= cp.multiply(b_ch_dis, C)
            - cp.multiply(z1, C)
            - C
            + cp.multiply(c_reg_share, C),
            # bc >= cp.multiply(b_ch_dis, C) - cp.multiply(z1, C_reg_down) - C + cp.multiply(c_reg_share, C_reg_down),
            ## Split net load in positive and negative for self.ToU and FiT charge
            nl
            == load
            - s_gen
            - b
            + cp.multiply(c_reg_share, ((Q_reg_down - Q_reg_up) / self.dt)),  # load
            nl == nl_pos + nl_neg,
            nl_pos <= cp.multiply(nl_pos_neg, nl_fuse),  # limits on net-load power
            -nl_neg
            <= cp.multiply(
                (-(nl_pos_neg - 1)), nl_fuse
            ),  # limits on net-load power fuse size
            -nl_pos <= 0,
            nl_neg <= 0,
        ]

        ## FCR-N constrains about battery availability
        for i in range(self.T):
            constraints += [
                q[i] <= Q_max - c_reg_share[i] * C * self.dt,
                q[i] >= Q_min + c_reg_share[i] * C * self.dt,
            ]

        ## Form objective
        costs = self.get_costs(params)
        log.debug(f"{device.name} FCRN price: {costs['FCR-N'].reshape(-1)} €/MW,h")
        log.debug(f"{device.name} wholesale price: {costs['ToU'].reshape(-1)} €/MWh")
        objective = cp.Minimize(
            ## ToU
            costs["ToU"].transpose() @ (nl_pos) * self.dt
            +  # costs["ToU"].transpose() @ (nl_pos - cp.multiply(c_reg_share, ((Q_reg_down - Q_reg_up) / self.dt))) * self.dt + \
            ## Battery operation
            costs["BESS"].transpose() @ (bd) * self.dt
            + costs["BESS"].transpose() @ (-bc) * self.dt
            + costs["BESS"].transpose()
            @ (cp.multiply((Q_reg_down + Q_reg_up), c_reg_share))  ## FiT
            + costs["FiT"].transpose()
            @ (nl_neg)
            * self.dt  # - costs["FiT"].transpose() @ (nl_neg-cp.multiply(c_reg_share, ((Q_reg_down - Q_reg_up) / self.dt))) * self.dt - \
            ## FCR-N
            - costs["FCR-N"].transpose() @ (cp.multiply(c_reg_share, C)) * self.dt  # +\
            # costs["ToU"].transpose() @ (cp.multiply(Q_reg_down, c_reg_share)) #+ \
            # costs["FiT"].transpose() @ (cp.multiply(-Q_reg_up, c_reg_share))
        )

        # # Form and solve problem
        problem = cp.Problem(objective, constraints)
        for variable in problem.variables():
            log.debug(
                f"{device.name} {variable.name()} variable value: {variable.value}"
            )
        log.info(f"Installed solvers: {cp.installed_solvers()}")
        # ['CPLEX', 'CVXOPT', 'ECOS', 'GLPK', 'GLPK_MI', 'OSQP', 'SCS']
        problem.solve(
            solver="GLPK_MI", verbose=True
        )  # , abstol=1e-2,reltol=1e-2,feastol=1e-2)
        log.info(f"{device.name} problem status: {problem.status}")
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            log.info(f"{device.name} optimal value: {problem.value}")
            for variable in problem.variables():
                log.debug(
                    f"{device.name} {variable.name()} variable value: {variable.value}"
                )
        else:
            log.error(f"{device.name} problem did not converged, trying another solver")
            # problem = cp.Problem(objective, constraints)
            # problem.solve(solver='ECOS_BB', verbose=True) #, max_iter=5000
            # if problem.status in ["infeasible", "unbounded"]:
            # log.error(f"{device.name} problem did not converged again, sending zeros")
            return (
                np.repeat(0.0, self.T + 1),
                np.repeat(0.0, self.T),
                np.repeat(0.0, self.T),
            )
            # raise ValueError("Problem did not converged")
            # for variable in problem.variables():
            #     log.debug(f"{device.name} {variable.name()} variable value: {variable.value}")
        return q.value, b.value, c_reg_share.value

    def get_costs(self, params) -> np.ndarray:
        """A simple method to get the cost array
        Returns:
            Array of costs
        Examples:
            >>> array = np.array([1, 1, 1 ,1, 1])
        """
        bess_oc = (1e3 * params["cost_euro_kwh"]) / (
            2
            * params["end_of_life"]
            * (params["up_bess_limit"] - params["low_bess_limit"])
        )
        t = np.linspace(1, self.T, num=self.T).reshape(self.T, 1)
        # u = 40*np.exp(0.3*np.cos((t+16)*np.pi/self.T)-0.3*np.cos((t+15)*np.pi/self.T) - \
        #     0.3*np.cos(t*4*np.pi/self.T))
        costs = {
            "BESS": np.repeat(bess_oc, self.T).reshape(self.T, 1),  # 93.75
            "FCR-N": self.get_forecasts(type="price", quantile="q50").reshape(
                self.T, 1
            ),
            "ToU": 1.24
            * self.get_forecasts(type="wholesale_price", quantile="q50").reshape(
                self.T, 1
            )
            + params["ToU_MWh_euro"],
            # np.repeat(params["ToU_MWh_euro"], self.T).reshape(self.T, 1), # u,
            "FiT": self.get_forecasts(type="wholesale_price", quantile="q50").reshape(
                self.T, 1
            )
            - params["FiT_MWh_euro"],
        }
        return costs
