import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


# -------------------- DSII -------------------- #


def load_data(file_name_list):
    master_df = pd.read_csv(file_name_list[0],nrows=60110)
    for files in file_name_list[1:]:
        df = pd.read_csv(files, nrows=60110)
        master_df = pd.concat([master_df, df], axis=1)
    fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum", "master_hr.csv")
    if os.path.isfile(fpf):
        pass
    else:
        master_df.to_csv("master_hr.csv")
    return master_df


# LOAD DATA
data = ['43711__Rain__hourly.csv',
        '43711__Sunshine__hourly.csv',
        '43711__Temperature__hourly.csv',
        '43711__Wind__hourly.csv',
        '43711__Pressure__hourly.csv',]

data_df = load_data(data)

# SAMPLE TEST SET FROM DATA

data_df['value_cat'] = np.ceil(data_df['Rainfall [mm]'])
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

for name in ["test_set_hr.csv", "train_set_hr.csv"]:
    fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum", name)
    if os.path.isfile(fpf):
        pass
    else:
        train_master.to_csv("train_set_hr.csv")
        test_master.to_csv("test_set_hr.csv")

train_master.info()
train_master = train_master.sort_index()
test_master = test_master.sort_index()

# -------------------- DSIII -------------------- #

from datetime import datetime


def drop_duplicate_cols(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df


train_master = drop_duplicate_cols(train_master)
test_master = drop_duplicate_cols(test_master)


def date_time_conversion(df, newcol):
    df['datetimes'] = pd.to_datetime(df[newcol])
    df['months'] = df['datetimes'].dt.month
    df['month names'] = df['datetimes'].dt.month_name()
    df['years'] = df['datetimes'].dt.year
    df['days'] = df['datetimes'].dt.dayofyear
    df['hour of day'] = df['datetimes'].dt.hour
    return df


train_master = date_time_conversion(train_master, 'Observation time UTC')
test_master = date_time_conversion(test_master, 'Observation time UTC')


def plot_rain_numfeat(df, x, y, plttype):
    if plttype.lower() == "line":
        rain_feat = sns.lineplot(data=df, x=x, y=y)
    elif plttype.lower() == "scatter":
        rain_feat = sns.scatterplot(data=df, x=x, y=y)
    elif plttype.lower() == 'hist':
        rain_feat = sns.histplot(data=df, x=x, y=y)
    else:
        rain_feat = sns.regplot(data=df, x=x, y=y)
    rain_feat.tick_params(axis='x', rotation=270)
    plt.show()


# plot_rain_numfeat(train_master, 'datetimes', 'Rainfall [mm]', "line")
# plot_rain_numfeat(train_master, 'Mean Relative Humidity [percent]', 'Rainfall [mm]', "line")
# plot_rain_numfeat(train_master,'Mean sea level pressure [Hpa]', 'Rainfall [mm]', "line")
# plot_rain_numfeat(train_master,'Station level pressure', 'Rainfall [mm]', "line")
# plot_rain_numfeat(train_master,'Speed [m/s]', 'Rainfall [mm]', "hist")
# plot_rain_numfeat(train_master, 'Mean Temperature [Deg C]', 'Rainfall [mm]', 'linreg')
#

# -------------------- DSIII.V: MAKING NEW FEATURES-------------------- #

def add_cloudiness(df, colname):
    clouds = 1 - np.array(df[colname])
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


train_master = add_cloudiness(train_master, 'Sunshine [hrs]')
train_master = add_seasons(train_master, 'months')
train_master = add_dewpoint(train_master, 'Mean Temperature [Deg C]', 'Mean Relative Humidity [percent]')
# train_master = add_enso(train_master, test_master)
# train_master = add_wind_direction(train_master, test_master,'43711__Wind__hourly.csv')
train_master = direction_encoding(train_master, 'Direction [deg T]')
train_master.to_csv("train_set_hr.csv")
fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum","train_set_hr.csv" )
if os.path.isfile(fpf):
    pass
else:
    train_master.to_csv("train_set_hr.csv")


test_master = add_cloudiness(test_master, 'Sunshine [hrs]')
test_master = add_seasons(test_master, 'months')
test_master = add_dewpoint(test_master,'Mean Temperature [Deg C]', 'Mean Relative Humidity [percent]')
# test_master = add_enso(test_master, train_master)
# test_master = add_wind_direction(test_master, train_master, '43711__Wind__hourly.csv')
test_master = direction_encoding(test_master, 'Direction [deg T]')
test_master.to_csv("test_set_hr.csv")
fpf = os.path.join("/Users/neilkumar/Desktop/Python/Oceanum","test_set_hr.csv" )
if os.path.isfile(fpf):
    pass
else:
    test_master.to_csv("test_set_hr.csv")

# -------------------- DSIII contd -------------------- #


def colormap(df, x, y, s, cmt, c=None, color=False):
    plot_df = pd.DataFrame({s:df[s],
                            x:df[x],
                            y:df[y]})
    plot_df = plot_df.fillna(plot_df[s].mean())
    if c is not None:
        c_plot = df[c]
    else:
        c_plot = None

    if color:
        color_plot = True
    else:
        color_plot = False
    plot_df.plot(kind='scatter',x=x,y=y,alpha=0.5,s=s,
                 c=c_plot,label=s,cmap=plt.get_cmap(cmt),
                 colorbar=color_plot, sharex=False)
    plt.show()


def plot_categorical(df, x, y, type):
    if type.lower() == "box":
        plot = sns.boxplot(data=df,x=x,y=y)
    elif type.lower() == "violin":
        plot = sns.violinplot(data=df, x=x, y=y)
    else:
        plot = sns.swarmplot(data=df,x=x,y=y)
    plot.set_title(y + " vs " + x)
    plt.show()


# plot_categorical(train_master, 'ENSO cat', 'Total_rainfall__mm', "violin")
# plot_categorical(train_master, 'Seasons', 'Total_rainfall__mm', "box")
# plot_rain_numfeat(train_master, 'cloudiness','Total_rainfall__mm',"scatter")
# plot_rain_numfeat(train_master, 'dewpoint (degC)','Total_rainfall__mm',"scatter")
#
# colormap(train_master,'Mean_wind_speed__m_s', 'Mean_9am_Humidity__percent',
#          'Total_rainfall__mm',"jet",'dewpoint (degC)',color=True)
# colormap(train_master,'Mean_air_temperature__Deg_C', 'Mean_9am_Humidity__percent',
#          'Total_rainfall__mm',"jet", c=None,color=False)
# colormap(train_master,'wind_cos', 'wind_sin','Total_rainfall__mm',"jet"
#          ,'Mean_wind_speed__m_s',True)


# -------------------- DSIV -------------------- #

# train_master_num = train_master.select_dtypes(include='number')
# train_master_cat = train_master.select_dtypes(exclude='number')
# test_master_num = test_master.select_dtypes(include='number')
# test_master_cat = test_master.select_dtypes(exclude='number')


def split_by_type(train, test):
    train_num = train.select_dtypes(include='number')
    train_cat = train.select_dtypes(exclude='number')
    test_num = test.select_dtypes(include='number')
    test_cat = test.select_dtypes(exclude='number')

    return train_num, train_cat, test_num, test_cat


# train_master_num, train_master_cat, test_master_num, test_master_cat = split_by_type(train_master, test_master)
# pass

def lag(train_num, test_num, lags, feats):
    for feat in feats:
        for lag in lags:
            train_num[f'{feat} lag: {str(lag)}'] = train_num[feat].shift(lag)
            test_num[f'{feat} lag: {str(lag)}'] = test_num[feat].shift(lag)
            train_num[f'{feat} lag: {str(lag)}'] = train_num[f'{feat} lag: {str(lag)}'].ffill()
            test_num[f'{feat} lag: {str(lag)}'] = test_num[f'{feat} lag: {str(lag)}'].ffill()
    return train_num, test_num


def rolling_avgs(train_num, test_num, windows, feats):
    for feat in feats:
        for window in windows:
            train_num[f"{window} hr rolling avg {feat}"] = train_num[feat].rolling(window=window, min_periods=1).mean().shift(1)
            test_num[f"{window} hr rolling avg {feat}"] = test_num[feat].rolling(window=window, min_periods=1).mean().shift(1)
            train_num[f"{window} hr rolling avg {feat}"] = train_num[f"{window} hr rolling avg {feat}"].ffill()
            test_num[f"{window} hr rolling avg {feat}"] = test_num[f"{window} hr rolling avg {feat}"].ffill()
    return train_num, test_num


def impute_all(train_num, train_cat, test_num, test_cat):
    from sklearn.impute import SimpleImputer
    num_imp = SimpleImputer(strategy='median')
    cat_imp = SimpleImputer(strategy='most_frequent')
    train_num = pd.DataFrame(num_imp.fit_transform(train_num),columns=train_num.columns,
                             index=train_num.index)
    train_cat = pd.DataFrame(cat_imp.fit_transform(train_cat), columns=train_cat.columns,
                             index=train_cat.index)
    test_num = pd.DataFrame(num_imp.transform(test_num), columns=test_num.columns,
                            index=test_num.index)
    test_cat = pd.DataFrame(cat_imp.transform(test_cat), columns=test_cat.columns,
                            index=test_cat.index)

    return train_num, train_cat, test_num, test_cat


# train_master_num, train_master_cat, test_master_num, test_master_cat = impute_all(train_master_num, train_master_cat,
#                                                                                   test_master_num, test_master_cat)


def drop_unwanted(train_num, train_cat, test_num, test_cat, feat_name):
    if feat_name in train_num.columns:
        train_num = train_num.drop(feat_name, axis=1)
    if feat_name in train_cat.columns:
        train_cat = train_cat.drop(feat_name, axis=1)
    if feat_name in test_num.columns:
        test_num = test_num.drop(feat_name, axis=1)
    if feat_name in test_cat.columns:
        test_cat = test_cat.drop(feat_name, axis=1)
    return train_num, train_cat, test_num, test_cat,


# train_master_num, train_master_cat, test_master_num, test_master_cat = drop_unwanted(train_master_num, train_master_cat,
#                                                                                      test_master_num, test_master_cat,
#                                                                                      "Months")
#
# train_master_num, train_master_cat, test_master_num, test_master_cat = drop_unwanted(train_master_num, train_master_cat,
#                                                                                      test_master_num, test_master_cat,
#                                                                                      "time"


def encoding(train_cat, test_cat):
    from sklearn.preprocessing import OneHotEncoder
    ohc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_cat = pd.DataFrame(ohc.fit_transform(train_cat),
                             columns=ohc.get_feature_names_out(),index=train_cat.index)
    test_cat = pd.DataFrame(ohc.transform(test_cat),
                            columns=ohc.get_feature_names_out(),index=test_cat.index)
    return train_cat, test_cat


# train_master_cat, test_master_cat = encoding(train_master_cat, test_master_cat)


def cyclic_scaling(train_num, test_num, col_name, periods):
    for period in periods:
        train_num[col_name + " sin"] = np.sin(2 * np.pi * train_num[col_name] / period)
        train_num[col_name + " cos"] = np.cos(2 * np.pi * train_num[col_name] / period)
        test_num[col_name + " sin"] = np.sin(2 * np.pi * test_num[col_name] / period)
        test_num[col_name + " cos"] = np.cos(2 * np.pi * test_num[col_name] / period)
    return train_num, test_num


# train_master_num, test_master_num = cyclic_scaling(train_master_num, test_master_num, 'month_num')


def variance(train_num,  test_num):
    from sklearn.feature_selection import VarianceThreshold
    low_var_remover = VarianceThreshold()
    train_num = pd.DataFrame(low_var_remover.fit_transform(train_num),
                             columns=train_num.columns, index=train_num.index)
    test_num = pd.DataFrame(low_var_remover.transform(test_num),
                             columns=test_num.columns, index=test_num.index)
    return train_num, test_num


# train_master_num, test_master_num = variance(train_master_num, test_master_num)


def stdscale(train_num, test_num, feat_list):
    from sklearn.preprocessing import StandardScaler
    train_num_scale = train_num[feat_list]
    test_num_scale = test_num[feat_list]
    scaler = StandardScaler()
    train_num_scale = pd.DataFrame(scaler.fit_transform(train_num_scale),
                                   columns=train_num_scale.columns,
                                   index=train_num_scale.index)
    test_num_scale = pd.DataFrame(scaler.transform(test_num_scale),
                                   columns=test_num_scale.columns,
                                   index=test_num_scale.index)
    train_num.update(train_num_scale)
    test_num.update(test_num_scale)
    return train_num, test_num


feats_to_scale = ["Station level pressure", "Mean sea level pressure [Hpa]", "Sunshine [hrs]", "cloudiness"]
# train_master_num, test_master_num = stdscale(train_master_num, test_master_num, feats_to_scale)


def combine(train_num, train_cat, test_num, test_cat):
    train_final = pd.merge(train_num, train_cat, left_index=True, right_index=True)
    test_final = pd.merge(test_num, test_cat, left_index=True, right_index=True)
    return train_final, test_final


# train_master, test_master = combine(train_master_num, train_master_cat, test_master_num, test_master_cat)


def pipeline(train, test, col_name_sc,drop_feat,feat_list, periods, lag_feat, rolling_feat):
    train_num, train_cat, test_num, test_cat = split_by_type(train, test)
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"Deficit [mm]")
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"Runoff [mm]")
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"Data source (water balance)")
    train_num, test_num = lag(train_num, test_num, [1,2,3,6,12,24], lag_feat)
    train_num, test_num = rolling_avgs(train_num, test_num, [3,6,12,24], rolling_feat)
    train_num, train_cat, test_num, test_cat = impute_all(train_num, train_cat, test_num, test_cat)
    for unwanted in drop_feat:
        train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat, unwanted)
    train_cat, test_cat = encoding(train_cat.drop('month names', axis=1), test_cat.drop('month names', axis=1))
    for col in col_name_sc:
        train_num, test_num = cyclic_scaling(train_num, test_num, col, periods)
    train_num, test_num = variance(train_num,  test_num)
    train_num, test_num = stdscale(train_num, test_num, feat_list)
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"hour of day")
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"months")
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"days")
    train_num, train_cat, test_num, test_cat = drop_unwanted(train_num, train_cat, test_num, test_cat,"month names")
    train_final, test_final = combine(train_num, train_cat, test_num, test_cat)

    return train_final, test_final


drop_list = ["Frequency [D/H]",
             "Data Source (rainfall)",
             "PERIOD [hrs]",
             "Data Source",
             "Data Source (max temp)",
             "Data Source (min temp)",
             "Grass Temperature [Deg C]",
             "Data Source (grass temp)",
             "Period (max temp)",
             "Period (min temp)",
             "Period (grass temp)",
             "Period (mean temp)",
             "Data Source (mean temp)",
             "Data Source (station level pressure)",
             "Direction [deg T]",
             "Data Source (mean sea level pressure)",
             "datetimes",
             "Deficit [mm]", "Runoff [mm]", "Data source (water balance)",
             "Observation time UTC",
             "Maximum Temperature [Deg C]", "Minimum Temperature [Deg C]", 'years'
             ]

lag_feat = ["Rainfall [mm]", "Mean sea level pressure [Hpa]", "Mean Relative Humidity [percent]",
            "dewpoint (degC)", "wind_sin","wind_cos", "cloudiness", "Sunshine [hrs]"]
roll_feat = lag_feat
train_master_final, test_master_final = pipeline(train_master, test_master,["months", "days", "hour of day"]
                                                 ,drop_list, feats_to_scale, [12, 365, 24],lag_feat, roll_feat)


# -------------------- DSV -------------------- #

X_train = train_master_final.drop('Rainfall [mm]', axis=1)
X_test = test_master_final.drop('Rainfall [mm]', axis=1)
y_train = train_master_final['Rainfall [mm]']
y_test = test_master_final['Rainfall [mm]']
y_train_bin = (y_train > 0).astype(int)
y_test_bin = (y_test > 0).astype(int)

X_train_sub = X_train.sample(n=1000)
y_train_sub = np.log1p(y_train.sample(n=1000))
print("target mean ",y_train_sub.mean())

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error


def mae_expm1(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return mean_absolute_error(y_true, y_pred)


mae_log = make_scorer(mae_expm1, greater_is_better=False)

ridge_score = cross_val_score(Ridge(), X_train_sub, y_train_sub, scoring=mae_log,cv=10,n_jobs=-1)
print((-ridge_score).mean())
print((-ridge_score).std())

rfr_score = cross_val_score(RandomForestRegressor(n_jobs=-1), X_train_sub, y_train_sub, scoring=mae_log, cv=10, n_jobs=-1)
print((-rfr_score).mean())
print((-rfr_score).std())

X_train_sub.columns = [str(col) for col in X_train_sub.columns]
X_train_sub.columns = X_train_sub.columns.str.replace(r'[\[\]<>]', '', regex=True)


xgb_score = cross_val_score(XGBRegressor(objective='reg:absoluteerror',
                                         n_estimators=100,
                                         learning_rate=0.1,
                                         max_depth=5,
                                         random_state=42), X_train_sub, y_train_sub, scoring=mae_log, n_jobs=-1, cv=10)
print((-xgb_score).mean())
print((-xgb_score).std())

svr_score = cross_val_score(SVR(), X_train_sub, y_train_sub,scoring=mae_log, n_jobs=-1, cv=10)
print((-svr_score).mean())
print((-svr_score).std())

pass


# -------------------- DSVI -------------------- #

from sklearn.model_selection import GridSearchCV

xgb_param_grid = {"n_estimators": [87, 90, 95, 100],
                  "learning_rate":[0.001, 0.005, 0.01, 0.1],
                  "max_depth":[11, 12, 13, 14, 15],
                  "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
                  "subsample":[0.6, 0.7, 0.8, 0.9],
                  "reg_alpha":[0.15, 0.16, 0.17, 0.1, 0.01]}
xgb_gs = GridSearchCV(XGBRegressor(), param_grid=xgb_param_grid,
                      cv=5,scoring=mae_log, n_jobs=-1)
xgb_gs.fit(X_train_sub, y_train_sub)
print("best params:", xgb_gs.best_params_)
print("best score:", xgb_gs.best_score_)


# -------------------- DSVII -------------------- #



