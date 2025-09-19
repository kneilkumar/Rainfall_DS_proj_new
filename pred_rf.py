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

data_df['value_cat'] = np.ceil(data_df['Total_rainfall__mm'] / 15)
data_df['value_cat'].where(data_df['value_cat'] < 5, 5.0, inplace=True)
# sns.histplot(data_df['value_cat'], kde=False)
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

strata = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69420)
for train_index, test_index in strata.split(data_df, data_df['value_cat']):
    train_master = data_df.loc[train_index]
    test_master = data_df.loc[test_index]

check_sampling = pd.DataFrame()
for samples in ("data_df", "train_master", "test_master"):
    sample_check = globals()[samples]["value_cat"].value_counts() / len(globals()[samples])
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
    new_time = pd.DataFrame({newcol + "new": new_times}, index=df.index)
    df = pd.concat([df, new_time], axis=1)
    df[newcol] = df['Month'] + ' ' + df['Year'].astype(str)
    return df


train_master = date_time_conversion(train_master, 'time')


def month_num_conversion(df, colname):
    df = pd.concat([df, pd.DataFrame({'month_num': [datetime.fromtimestamp(timestamp).month
                                                                    for timestamp in list(df[colname])]},
                                                                    index=df.index)], axis=1)
    return df


train_master = month_num_conversion(train_master, 'timenew')
train_master = train_master.sort_index()


def plot_rain_numfeat(df, x, y, plttype):
    if plttype.lower() == "line":
        rain_feat = sns.lineplot(data=df, x=x, y=y)
    elif plttype.lower() == "scatter":
        rain_feat = sns.scatterplot(data=df, x=x, y=y)
    else:
        rain_feat = sns.regplot(data=df, x=x, y=y)
    rain_feat.tick_params(axis='x', rotation=270)
    plt.show()


# plot_rain_numfeat(train_master, 'time', 'Total_rainfall__mm', "line")
# plot_rain_numfeat(train_master, 'Mean_9am_Humidity__percent', 'Total_rainfall__mm', "line")
# plot_rain_numfeat(train_master,'Mean_sea-level_pressure_at_9am__hPa', 'Total_rainfall__mm', "line")
# plot_rain_numfeat(train_master,'Mean_vapour_pressure__hPa', 'Total_rainfall__mm', "line")
# plot_rain_numfeat(train_master,'Mean_wind_speed__m_s', 'Total_rainfall__mm', "scatter")
# plot_rain_numfeat(train_master, 'Mean_air_temperature__Deg_C', 'Total_rainfall__mm', 'linreg')


rain_temp_humid = pd.DataFrame({"Rainfall (mm)": train_master['Total_rainfall__mm'],
                                "Temperature (C)": train_master['Mean_air_temperature__Deg_C'],
                                "Humidity (%)": train_master['Mean_9am_Humidity__percent']})
rain_temp_humid = rain_temp_humid.fillna(rain_temp_humid['Rainfall (mm)'].mean())
rain_temp_humid.plot(kind='scatter', x='Temperature (C)', y='Humidity (%)',
                     alpha=0.5, s="Rainfall (mm)",
                     label="Rainfall (mm)", cmap=plt.get_cmap("jet"),
                     colorbar=False, sharex=False)
# plt.show()


rain_vap_sea = pd.DataFrame({"Rainfall (mm)": train_master['Total_rainfall__mm'],
                             "Vapour Pressure (hPa)": train_master['Mean_vapour_pressure__hPa'],
                             "Sea Level Pressure (hPa)": train_master['Mean_sea-level_pressure_at_9am__hPa']})
rain_vap_sea = rain_vap_sea.fillna(rain_vap_sea['Rainfall (mm)'].mean())
rain_vap_sea.plot(kind='scatter',x='Sea Level Pressure (hPa)',y='Vapour Pressure (hPa)',
                  alpha=0.5,s="Rainfall (mm)", label="Rainfall (mm)",cmap=plt.get_cmap("jet"),
                  colorbar=False, sharex=False)
# plt.show()


# -------------------- DSIII.V: MAKING NEW FEATURES-------------------- #

def add_cloudiness(df, colname):
    clouds = [720 - vals
                        if vals in [4,6,9,11] else
                        744 - vals
                        if vals in [1,3,5,7,8,10,12] else
                        725.5 - vals for vals in np.array(df[colname])]
    cloudiness = pd.DataFrame({'cloudiness': clouds}, index=df.index)
    df = pd.concat([df, cloudiness], axis=1)
    return df


def add_seasons(df, colname):
    szn_list = ["Summer" if vals in [12, 1, 2] else
                     "Autumn" if vals in [3,4,5] else
                     "Winter" if vals in [6,7,8] else
                     "Spring" for vals in list(df[colname])]
    szn = pd.DataFrame({'Seasons':szn_list}, index=df.index)
    df = pd.concat([df, szn], axis=1)
    return df


def add_enso(df_include, df_exclude):
    enso = pd.read_csv('ENSO_Oct2018_Aug2025.csv')
    test_data = list(df_exclude.index)
    enso_use = enso.drop(test_data)
    index = list(enso_use['Value'])
    cats = ["La Nina" if p > 0.5 else
            "El Nino" if p < -0.5 else
            "Neutral" for p in index]
    cats_df = pd.DataFrame({"ENSO cat": cats,
                            "ENSO num": enso_use['Value']}, index=df_include.index)
    df_include = pd.concat([df_include, cats_df], axis=1)
    return df_include


def add_dewpoint(df, col1, col2):
    temp = np.array(df[col1])
    rh = np.array(df[col2])
    gamma = np.log(rh/100) + ((17.625*temp)/(243.04 + temp))
    dewpoint = (243.04*gamma)/(17.625 - gamma)
    dp_df = pd.DataFrame({"dewpoint (degC)": dewpoint}, index=df.index)
    df = pd.concat([df, dp_df], axis=1)
    return df


def add_wind_direction(df, exclude, file):
    wind_dir_hrly = pd.read_csv(file)
    wind_dir_hrly['Observation time UTC'] = pd.to_datetime(wind_dir_hrly['Observation time UTC'])
    wind_dir_hrly['YEAR_MONTH'] = wind_dir_hrly['Observation time UTC'].dt.strftime('%Y.%m')
    wind_dir_hrly['u_component'] = np.cos(wind_dir_hrly['Direction [deg T]']*np.pi/180)
    wind_dir_hrly['v_component'] = np.sin(wind_dir_hrly['Direction [deg T]']*np.pi/180)
    drop_index = wind_dir_hrly[wind_dir_hrly['YEAR_MONTH'].astype(float) == 2018.09].index
    wind_dir_hrly.drop(drop_index, inplace=True)
    ym = wind_dir_hrly['YEAR_MONTH'].unique()
    u_months_avg = []
    v_months_avg = []
    for months in ym:
        condition = wind_dir_hrly['YEAR_MONTH'] == months
        u_month = wind_dir_hrly[condition]['u_component']
        u_month_avg = np.sum(u_month)/len(u_month)
        u_months_avg.append(u_month_avg)

    for months in ym:
        condition = wind_dir_hrly['YEAR_MONTH'] == months
        v_month = wind_dir_hrly[condition]['v_component']
        v_month_avg = np.sum(v_month)/len(v_month)
        v_months_avg.append(v_month_avg)

    u_months_avg = np.array(u_months_avg)
    v_months_avg = np.array(v_months_avg)

    test_data = list(exclude.index)
    wind_dir = pd.DataFrame({"wind_direction": (np.degrees(np.arctan2(u_months_avg, v_months_avg))+360) % 360,
                             "Months": ym}, index=data_df.index)
    wind_dir = wind_dir.drop(test_data)
    df = pd.concat([df, wind_dir], axis=1)
    return df


def direction_encoding(df, colname):
    df['wind_sin'] = np.sin(np.deg2rad(df[colname]))
    df['wind_cos'] = np.cos(np.deg2rad(df[colname]))
    return df


train_master = add_cloudiness(train_master, 'Total_sunshine_hours__Hrs')
train_master = add_seasons(train_master, 'month_num')
train_master = add_dewpoint(train_master, 'Mean_air_temperature__Deg_C', 'Mean_9am_Humidity__percent')
train_master = add_enso(train_master, test_master)
train_master = add_wind_direction(train_master, test_master,'43711__Wind__hourly.csv')
train_master = direction_encoding(train_master, 'wind_direction')
train_master.to_csv("train_set_master.csv")
fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum","train_set_master.csv" )
if os.path.isfile(fpf):
    pass
else:
    train_master.to_csv("train_set_master.csv")


# -------------------- DSIII contd -------------------- #

rain_speed_dir = pd.DataFrame({"Rainfall (mm)": train_master['Total_rainfall__mm'],
                             "Wind cos (rad)": train_master['wind_cos'],
                             "Wind sin (rad)": train_master['wind_sin']})
rain_speed_dir = rain_speed_dir.fillna(rain_speed_dir['Rainfall (mm)'].mean())
rain_speed_dir.plot(kind='scatter',x='Wind cos (rad)',y='Wind sin (rad)',
                  alpha=0.5,s="Rainfall (mm)", c=train_master['Mean_wind_speed__m_s'],
                  label="Rainfall (mm)",cmap=plt.get_cmap("jet"),
                  colorbar=True, sharex=False)
# plt.show()

rain_speed_hum = pd.DataFrame({"Rainfall (mm)": train_master['Total_rainfall__mm'],
                             "Wind speed": train_master['Mean_wind_speed__m_s'],
                             "Wind humidity": train_master['Mean_9am_Humidity__percent']})
rain_speed_hum = rain_speed_hum.fillna(rain_speed_hum['Rainfall (mm)'].mean())
rain_speed_hum.plot(kind='scatter',x='Wind speed',y='Wind humidity',
                  alpha=0.5,s="Rainfall (mm)",c=train_master["ENSO num"],
                  label="Rainfall (mm)",cmap=plt.get_cmap("jet"),
                  colorbar=True, sharex=False)
# plt.show()


def colormap(df, x, y, s, c=None, color=False):
    plot_df = pd.DataFrame({s:df[s],
                            x:df[x],
                            y:df[y]})
    plot_df = plot_df.fillna(plot_df[s].mean())
    if c is not None:
        c_plot = c
    else:
        c_plot = None

    if color:
        color_plot = True
    else:
        color_plot = False
    plot_df.plot(kind='scatter',x=x,y=y,alpha=0.5,s=s,
                 c=c_plot,label=y,cmap=plt.get_cmap("jet"),
                 colorbar=color_plot, sharex=False)


def plot_categorical(df, x, y, type):
    if type.lower() == "box":
        plot = sns.boxplot(data=df,x=x,y=y)
    elif type.lower() == "violin":
        plot = sns.violinplot(data=df, x=x, y=y)
    else:
        plot = sns.swarmplot(data=df,x=x,y=y)
    plot.set_title(y + " vs " + x)
    plt.show()


plot_categorical(train_master, 'ENSO cat', 'Total_rainfall__mm', "violin")
plot_categorical(train_master, 'Seasons', 'Total_rainfall__mm', "box")
plot_rain_numfeat(train_master, 'cloudiness','Total_rainfall__mm',"scatter")
plot_rain_numfeat(train_master, 'dewpoint (degC)','Total_rainfall__mm',"scatter")