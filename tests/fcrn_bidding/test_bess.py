import csv
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

if __name__ == "__main__":
    requests.get("`http://localhost:5080/enable/api_control")
    requests.get("`http://localhost:5080/keep_alive")
    # requests.get('`http://localhost:5080/set/fixed_ap_ref/5')
    requests.get("`http://localhost:5080/set/p_ref/16")
    # requests.get('`http://localhost:5080/set/soc_limiter/min/0')
    # requests.get('`http://localhost:5080/set/soc_limiter/max/100')
    # requests.get('`http://localhost:5080/set/api_control_task/FCRN')
    requests.get("`http://localhost:5080/set/api_control_task/discharge")
    print(requests.get("`http://localhost:5080/get/api_control").json())
    print(requests.get("`http://localhost:5080/get/soc").json())
    print(requests.get("`http://localhost:5080/get/p_ref").json())
    print(requests.get("`http://localhost:5080/get/soc_limiter/min").json())
    print(requests.get("`http://localhost:5080/get/soc_limiter/max").json())
    print(
        requests.get("`http://localhost:5080/get/soc_limiter/available_capacity").json()
    )
    print(requests.get("`http://localhost:5080/get/api_control_allowed_tasks").json())
    print(requests.get("`http://localhost:5080/get/api_control_task").json())
    # print(requests.get('`http://localhost:5080/get/fixed_ap_ref').json())
    # print(requests.get('`http://localhost:5080/get/fixed_rp_ref').json())
    # /set/api_control_task/fixed_reference
    # /set/fixed_ap_ref/5
    # /set/fixed_rp_ref/0
    # /get/fixed_ap_ref
    # /get/fixed_rp_ref
    # requests.get('`http://localhost:5080/disable/api_control')
