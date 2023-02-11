import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from __main__ import logger_name
from dateutil import tz
from dateutil.relativedelta import relativedelta

log = logging.getLogger(logger_name)
import functools


# This decorator can be applied to
def with_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log.info(f"LOG: Running job {func.__name__}")
        result = func(*args, **kwargs)
        log.info(f"LOG: Job {func.__name__} completed")
        return result

    return wrapper


def reindex_dataframe(df, method="ffill", freq="15T", delta="1H"):
    """
    Adds one hour to the last row and fills forward
    """
    df = df.reindex(
        pd.date_range(
            df.index.min(),
            df.index.max() + pd.Timedelta(delta),
            freq=freq,
            closed="left",
        ),
        method=method,
    )
    return df


def round_time(tm, minutes=15):
    """ """
    # tm = datetime.utcnow()
    discard = timedelta(
        minutes=tm.minute % minutes, seconds=tm.second, microseconds=tm.microsecond
    )
    tm -= discard
    if discard >= timedelta(minutes=minutes / 2.0):
        tm += timedelta(minutes=minutes)
    return tm


def round_results(array, decimals=3):
    """
    Rounds results with N decimals
    """
    return np.array(np.around(np.array(array), decimals))


def convert_kW_to_MW(array):
    """
    Converts kW to MW
    """
    return array * 0.001


def convert_MW_to_kW(array):
    """
    Converts MW to kW
    """
    return array * 1e3


def convert_to_euro(array):
    """
    Converts cents to euro
    """
    return array * 100.0


def convert_to_utc(local_datetime):
    """
    Converts local datetime to UTC format
    """
    # from_zone = tz.gettz("Europe/Helsinki")
    # to_zone = tz.gettz("UTC")
    from_zone = tz.tzlocal()
    to_zone = tz.tzutc()
    return local_datetime.replace(tzinfo=from_zone).astimezone(to_zone)


def convert_from_utc_to_local(utc_datetime):
    """
    Converts local datetime to UTC format
    """
    # from_zone = tz.gettz("Europe/Helsinki")
    # to_zone = tz.gettz("UTC")
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    return utc_datetime.replace(tzinfo=from_zone).astimezone(to_zone)


def convert_to_strf(datetime_input):
    """
    Converts datetime to strftime format
    """
    return datetime_input.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_offset(time, offset_full):
    if not offset_full is None:
        for offset in offset_full.split():
            if "D" in offset:
                time -= timedelta(days=int(offset.replace("D", "")))
                # end_time -= timedelta(days=int(offset.replace("D", "")))
            if "M" in offset:
                time += relativedelta(months=-int(offset.replace("M", "")))
                # end_time += relativedelta(months=-int(offset.replace("M", "")))
            if "Y" in offset:
                time += relativedelta(years=-int(offset.replace("Y", "")))
                # end_time += relativedelta(years=-int(offset.replace("Y", "")))
    return time


def get_start_end_time(query_type, offset=None):
    """
    Returns time period based on query type
    """
    start_hour = 22
    if query_type == "last_15_minutes":
        end_time = datetime.utcnow().replace(second=0, microsecond=0)
        end_time = validate_offset(end_time, offset)
        start_time = end_time - timedelta(minutes=15)
    elif query_type == "last_60_minutes":
        end_time = datetime.utcnow().replace(second=0, microsecond=0)
        end_time = validate_offset(end_time, offset)
        start_time = end_time - timedelta(minutes=60)
    elif query_type == "last_day":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=1)
    elif query_type == "next_day":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=1)
    elif query_type == "next_15_minutes":
        start_time = datetime.utcnow().replace(second=0, microsecond=0)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(minutes=15)
    elif query_type == "next_60_minutes":
        start_time = datetime.utcnow().replace(second=0, microsecond=0)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(minutes=60)
    elif query_type == "now_till_next_day":
        st = datetime.utcnow().replace(second=0, microsecond=0)
        start_time = (
            pd.Series(st).dt.round(f"15min").at[0]
        )  # round_time(datetime.utcnow())
        start_time = validate_offset(start_time, offset)
        end_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
    elif query_type == "day_before":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=1 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=1)
    elif query_type == "last_week":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=7 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(7 + 1))
    elif query_type == "last_two_weeks":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=14 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(14 + 1))
    elif query_type == "last_four_weeks":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=28 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(28 + 1))
    elif query_type == "last_two_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 2 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 2 + 1))
    elif query_type == "last_three_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 3 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 3 + 1))
    elif query_type == "last_six_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 6 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 6 + 1))
    elif query_type == "two_months_before_two_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 4 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 2 + 1))
    elif query_type == "two_months_before_four_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 6 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 2 + 1))
    elif query_type == "month_before_two_months":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=30 * 3 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(30 * 2 + 1))
    elif query_type == "last_eight_weeks":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=2 * 28 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(2 * 28 + 1))
    elif query_type == "last_year":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=365 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(365 + 1))
    elif query_type == "last_three_years":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=3 * 365 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(3 * 365 + 1))
    elif query_type == "last_five_years":
        start_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=5 * 365 + 1)
        start_time = validate_offset(start_time, offset)
        end_time = start_time + timedelta(days=(5 * 365 + 1))
    elif query_type == "all_history":
        start_time = datetime.strptime(
            f"2013-01-02T{start_hour}:00:00Z", "%Y-%m-%dT%H:%M:%SZ"
        )
        end_time = datetime.utcnow().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
    else:
        log.error(f"query_type={query_type} does not exist.")
    return start_time, end_time


def log_level(name) -> int:
    """
    Returns log level number based on its name
    """
    levels = {
        "DEBUG": 10,  #  Detailed information, for diagnosing problems. Value=10.
        "INFO": 20,  # Confirm things are working as expected. Value=20.
        "WARNING": 30,  # Something unexpected happened, or indicative of some problem. But the software is still working as expected. Value=30.
        "ERROR": 40,  # More serious problem, the software is not able to perform some function. Value=40
        "CRITICAL": 50,
    }  # A serious error, the program itself may be unable to continue running. Value=50
    return levels[name]
