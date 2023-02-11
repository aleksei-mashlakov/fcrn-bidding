from datetime import datetime, timedelta

import mysql.connector as mariadb
import pandas as pd

mariadb_connection = mariadb.connect(
    host="`http://localhost",
    port="43306",
    user="grafana",
    password="xxxxxxxx",
    database="xxxxxxxx",
)
cursor = mariadb_connection.cursor()
# retrieving information
start_time = datetime.utcnow() - timedelta(seconds=1900)

end_time = datetime.utcnow()
start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
print(start_time)
end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
try:
    cursor.execute(
        f"SELECT P_BESS_AVG, TIMESTAMP FROM BESS_MC WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'"
    )
except mariadb.Error as e:
    print(f"MAriaDB Error: {e}")
result = cursor.fetchall()
data = []
for row in result:
    data.append(row)
cursor.close()
df = pd.DataFrame(data, columns=["power_activated", "Time"]).set_index("Time")
print(df)
