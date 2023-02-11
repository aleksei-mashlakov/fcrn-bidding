import cvxpy as cp
import numpy as np
import requests

# Here we plot the demands u and prices p.
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(1)
# T = 96
# t = np.linspace(1, T, num=T).reshape(T,1)
# p = np.exp(-np.cos((t-15)*2*np.pi/T)+0.01*np.random.randn(T,1))
# u = 2*np.exp(-0.6*np.cos((t+40)*np.pi/T) - \
#     0.7*np.cos(t*4*np.pi/T)+0.01*np.random.randn(T,1))
# p = p
# u = u
# plt.figure(1)
# plt.plot(t/4, p, 'g', label=r"$p$");
# plt.plot(t/4, u, 'r', label=r"$u$");
# plt.ylabel("$")
# plt.xlabel("t")
# plt.legend()
# plt.show()

headers = {
    "Accept": "text/csv",
    "x-api-key": "XWN3lhIccU6MecQaUwbiq3zu4yNqAevA5OCJQ0N0",
}

params = (
    ("start_time", "2021-01-11T22:00:00+00:00"),
    ("end_time", "2021-01-21T23:00:00+00:00"),
)
#
# response = requests.get('https://api.fingrid.fi/v1/variable/%s/events/csv' % (str(79)), headers=headers, params=params)
# import pandas as pd
# import datetime
# resp = [line.decode('utf-8').split(',') for line in response.iter_lines()]
# df = pd.DataFrame(data=resp[1:], columns=resp[0]).drop('end_time', axis=1)
# df['start_time'] = pd.to_datetime(df['start_time'])
# #+ datetime.timedelta(minutes=5) #pd.DatetimeIndex(pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
# df.set_index('start_time', inplace=True)
# df.index.rename('timestamp', inplace=True)
# df[['value']] = df[['value']].astype(float)
# print(df)
# print(df.shape)
# print(df.dtypes)


# df.reset_index().to_csv('./data/raw/fcrn_hourly_prices.csv', index=False)


# df = pd.read_csv(response)
# print(df.head())
# print(response.raw())
# NB. Original query string below. It seems impossible to parse and
# reproduce query strings 100% accurately so the one below is given
# in case the reproduced version is not "correct".
# response = requests.get('https://api.fingrid.fi/v1/variable/79/events/csv?start_time=2019-09-08T01%3A00%3A00%2B03%3A00&end_time=2019-09-09T01%3A00%3A00%2B03%3A00', headers=headers)

import influxdb
import mxnet
from gluonts.dataset.common import ListDataset, load_datasets
from gluonts.dataset.field_names import FieldName
from influxdb import DataFrameClient, InfluxDBClient

influxdb = DataFrameClient(host="", port=8086, username="", password="", database="")

# influxdb2 = DataFrameClient(
#     host="",
#     port=8086,
#     username="",
#     password="",
#     database=""
# )
#
#
# influxdb3 = DataFrameClient(
#     host="",
#     port=,
#     username="",
#     password="",
#     database=""
# )


def make_query(
    field_name="price",
    measurement_name="nordicpower",
    tag="FCR-N",
    start_time="2012-12-22T00:00:00Z",
    end_time="2013-01-01T00:00:00Z",
    time_step="60m",
    tag_name="Type",
    agg="max",
):
    """A wrapper function to generate string clauses for InfluxDBClient"""
    select_clause = (
        'SELECT {}("{}") FROM "{}" '
        "WHERE \"{}\"='{}' AND time >= '{}' AND time <= '{}' "
        'GROUP BY time({}), "{}" fill(linear)'
    ).format(
        agg,
        field_name,
        measurement_name,
        tag_name,
        tag,
        start_time,
        end_time,
        time_step,
        tag_name,
    )
    return select_clause


# influxdb.query('DROP \"q10\" FROM \"nordicpower\"' \
#                 "WHERE Type=\'FCR-N\'")# AND time >= \'2020-02-01T00:00:00Z\'")
# influxdb.query('DROP \"q10\" FROM \"dominoes\"' \
#                 "WHERE Type=\'FCR-N\'")# AND time >= \'2020-02-01T00:00:00Z\'")
# print('sctring')
# print(influxdb.query('SELECT Power::string FROM nordicpower'))
# print(influxdb.get_list_series(measurement='nordicpower',database='nordicpower'))
#
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'FCR-N price')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'bess test')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'submitted')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'accepted')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'FCR-N up_power')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'FCR-N down_energy')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'FCR-N up_energy')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'FCR-N down_power')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'LUT load')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'q10')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'q25')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'q50')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'q75')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'q90')))
# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("nordicpower", 'amr_kw')))


# field = "price"
# print("Writing {} data to {} field".format('dominoes', field))
# for column_name in list(df.columns.astype(str)):
#     print(df[[column_name]].rename(columns={column_name:field}))
#     influxdb.write_points(df[[column_name]].rename(columns={column_name:field}),
#                           measurement='nordicpower',
#                           field_columns=[field],
#                           tags={#'global_tag':'Type',
#                                 'Type':'FCR-N'},
#                           protocol = 'line',
#                           batch_size=10000
#                           )


# querry = make_query(field_name="frequency",
#                     measurement_name="frequency",
#                     tag='Fingrid frequency measurement',
#                     start_time='2015-01-01T00:00:00Z',
#                     end_time='2015-02-01T00:00:00Z',
#                     time_step='1m',
#                     tag_name="name",
#                     agg='mean')

# querry = make_query(field_name="kwh",
#                     measurement_name="solar_forecast",
#                     tag="Lappeenranta",
#                     start_time='2020-10-04T00:00:00Z',
#                     end_time='2020-10-05T00:00:00Z',
#                     time_step='1h',
#                     tag_name="location",
#                     agg='mean')
# print(querry)
# print(influxdb2.query(querry))
# print(influxdb2.get_list_continuous_queries())
# print(influxdb2.get_list_measurements())
# print(influxdb2.get_list_series(database='bcdc', measurement='solar_forecast', tags=None))


# print(influxdb.get_list_measurements())
# print("Read DataFrame")
# print(querry)
# print(influxdb2.query(querry))#'SELECT \"Price\" from \"nordicpower\"'))
# print(influxdb.query('SELECT \"Price\" from \"nordicpower\"')['nordicpower'].head())
# influxdb.close()
# from datetime import datetime, timedelta
#
# print(datetime.utcnow())
# print(datetime.utcnow().replace(hour=23, minute=0, second=0, microsecond=0))
# print(datetime.utcnow().replace(hour=23, minute=0, second=0, microsecond=0) - timedelta(days=1))
# for minutes in range(0, 60, 15):
#     print(minutes)
#
# dt = datetime.now()
# discard = timedelta(minutes=dt.minute % 15,
#                     seconds=dt.second,
#                     microseconds=dt.microsecond)
# dt -= discard
# if discard >= timedelta(minutes=7.5):
#     dt += timedelta(minutes=15)
# while datetime.now() < next_step_at:
#     time.sleep(1)

# def make_query(field_name="price", measurement_name="nordicpower", #tag_name="Type",
#                tags={"Type":'FCR-N'}, start_time='2020-11-02T00:00:00Z',
#                end_time='2020-11-03T00:00:00Z',
#                freq='15m', agg='max', scale=1) -> str:
#     """
#         A wrapper function to generate string clauses for InfluxDBClient
#     """
#     # make condition string on tags
#     tag = ""
#     tag_name = ''
#     for k, v in tags.items():
#         tag += "\"{}\"=\'{}\' AND ".format(k, v)
#         tag_name += ', \"{}\"'.format(k)
#
#     # make full request string
#     select_clause = ('SELECT {}(\"{}\")*{} FROM \"{}\" ' \
#                      "W# print(influxdb.query('DELETE FROM \"{}\" WHERE Type=\'{}\' '.format("dominoes", 'bess test')))
# HERE {} time >= \'{}\' AND time < \'{}\' " \
#                      'GROUP BY time({}){} fill(linear)'
#                      ).format(agg, field_name, scale, measurement_name, tag,
#                               start_time, end_time, freq, tag_name)
#     print('Requesting data: {}'.format(select_clause))
#     return select_clause
#
# print("".join({"Type":'FCR-N'}.values()))
# print(str({"Type":'FCR-N'}.values()))
# print(type({"Type":'FCR-N'}.values()))

# print(influxdb.query(make_query()))
