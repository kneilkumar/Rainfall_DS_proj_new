import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()


# -------------------- DSII -------------------- #


def load_data(file_name_list):
    master_df = pd.DataFrame()
    i = 17
    time_file = pd.read_csv(file_name_list[0])
    master_df['Month'] = time_file['PERIOD']
    master_df['Year'] = time_file['YEAR']
    for files in file_name_list:
        while files[i] != '.':
            i += 1
        feat_name = files[16:i]
        file_df = pd.read_csv(files)
        master_df[feat_name] = file_df['STATS_VALUE']
        i = 17
    full_path_file = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum", "data_master_copy.csv")
    if os.path.isfile(full_path_file):
        pass
    else:
        master_df.to_csv("data_master_copy.csv")
    return master_df


# LOAD DATA

data = ['43711__monthly__Mean_9am_Humidity__percent.csv',
        '43711__monthly__Mean_air_temperature__Deg_C.csv',
        '43711__monthly__Mean_sea-level_pressure_at_9am__hPa.csv',
        '43711__monthly__Mean_vapour_pressure__hPa.csv',
        '43711__monthly__Mean_wind_speed__m_s.csv',
        '43711__monthly__Total_rainfall__mm.csv',
        '43711__monthly__Total_sunshine_hours__Hrs.csv']

data_df = load_data(data)

# SAMPLE TEST SET FROM DATA

data_df['value_cat'] = np.ceil(data_df['Total_rainfall__mm']/15)
data_df['value_cat'].where(data_df['value_cat'] < 5, 5.0, inplace=True)
# sns.histplot(data_df['value_cat'], kde=False)
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

strata = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69420)
for train_index, test_index in strata.split(data_df,data_df['value_cat']):
    train_master = data_df.loc[train_index]
    test_master = data_df.loc[test_index]


check_sampling = pd.DataFrame()
for samples in ("data_df", "train_master", "test_master"):
    sample_check = globals()[samples]["value_cat"].value_counts()/len(globals()[samples])
    check_sampling[samples] = sample_check
check_sampling = check_sampling.sort_index()
print(check_sampling.head())

for val in (train_master, test_master):
    val.drop(["value_cat"], axis=1, inplace=True)


for name in ["test_set_master.csv", "train_set_master.csv"]:
    fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum", name)
    if os.path.isfile(fpf):
        pass
    else:
        train_master.to_csv("train_set_master.csv")
        test_master.to_csv("test_set_master.csv")


train_master.info()

# -------------------- DSIII -------------------- #

from datetime import datetime


def date_time_conversion(df, newcol):
    df[newcol] = df['Month'] + df['Year'].astype(str)
    old_times = list(df[newcol])
    new_times = [datetime.strptime(time, "%B%Y").timestamp() for time in old_times]
    new_time = pd.DataFrame({newcol+"new": new_times}, index=df.index)
    df = pd.concat([df, new_time], axis=1)
    df[newcol] = df['Month'] + ' ' + df['Year'].astype(str)
    return df


train_master = date_time_conversion(train_master, 'time')


train_master = train_master.sort_index()
rain_time = sns.lineplot(data=train_master, x='time', y='Total_rainfall__mm')
# rain_time.tick_params(axis='x', rotation=270)
# plt.show()
#
# rain_humidity = sns.scatterplot(data=train_master,x='Mean_9am_Humidity__percent', y='Total_rainfall__mm')
# plt.show()
#
# rain_temp = sns.scatterplot(data=train_master, x='Mean_air_temperature__Deg_C', y='Total_rainfall__mm')
# plt.show()

rain_temp_humid = pd.DataFrame({"Rainfall (mm)": train_master['Total_rainfall__mm'],
                                "Temperature (C)": train_master['Mean_air_temperature__Deg_C'],
                                "Humidity (%)": train_master['Mean_9am_Humidity__percent']})
rain_temp_humid = rain_temp_humid.fillna(rain_temp_humid['Rainfall (mm)'].mean())
rain_temp_humid.plot(kind='scatter', x='Temperature (C)', y='Humidity (%)',
                     alpha=0.1, s=train_master['timenew']/1e7, c="Rainfall (mm)",
                     label="time", cmap=plt.get_cmap("jet"),
                     colorbar=True, sharex=False,)
plt.show()


