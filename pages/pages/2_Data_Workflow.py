import datetime

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import altair as alt
import plotly.graph_objects as go
from pandas import wide_to_long
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMapWithTime
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium, folium_static
import plotly.express as px


st.write("# Data Workflow Demo")
st.subheader(body=" - How I got to my final predictions",divider="rainbow")

mangere_df = pd.read_csv("csv_mangere_for_dash")
feat_eng_mangere = pd.read_csv("feat_eng_df.csv")

st.subheader("Feature Engineering: Dataframe Comparisons (uses Mangere as example)")
before_tab, after_tab = st.tabs(["Before", "After"])
with before_tab:
    st.markdown("**Pre-Featuring Engineering**")
    st.markdown("- Lots of unnecessary columns - adds noise ")
    st.markdown("- Lack of features representing system-element interaction (e.g. windspeed x humidity)")
    st.markdown("- No imputation or removal of missing data")
    st.dataframe(mangere_df.drop(["classifications", "regressionss", "final_preds"], axis=1))
with after_tab:
    st.markdown("**Post-Featuring Engineering**")
    st.markdown("- Scaling and encoding features applied (e.g. standard scaling and one hot encoding)")
    st.markdown("- Rolling averages and lag features included for range of periods")
    st.markdown("- Data cleaning prcedures applied (e.g. dropping of NaN columns)")
    st.dataframe(feat_eng_mangere)

st.subheader("Feature Engineering: Feature Comparison")
st.markdown("We can compare the effect that manipulated features have compared to raw features when it comes to "
            "predicting rainfall.  Let's compare all the hours from 04/04/2022 to 04/04/2023")
st.markdown("**Taking Pressure, Wind Speed and Pressure x Wind Speed (horizontal component) as examples:**")

pressure, wind_cos, presscos = st.tabs(["Pressure", "Wind", "Pressure x Wind (Cos)"])

man_sub = mangere_df.iloc[6058:7825]
temp = man_sub["Mean Temperature [Deg C]"]
hum = man_sub["Mean Relative Humidity [percent]"]
f_e_sub = feat_eng_mangere

temphum = f_e_sub["temphum"]
rain = f_e_sub["Rainfall [mm]"]

with pressure:
    st.markdown("Pressure vs Rainfall")
    fig = px.scatter(data_frame=man_sub,x="Mean sea level pressure [Hpa]", y="Rainfall [mm]")
    event = st.plotly_chart(fig, key="raw_temp", on_select="rerun")

with wind_cos:
    st.markdown("Wind vs Rainfall")
    fig1 = px.scatter(data_frame=man_sub,x="Speed [m/s]", y="Rainfall [mm]")
    event2 = st.plotly_chart(fig1, key="raw_hum", on_select="rerun")

with presscos:
    st.markdown("Pressure Trend vs Rainfall")
    fig2 = px.scatter(data_frame=f_e_sub,x="pressure trend cos", y="Rainfall [mm]")
    event3 = st.plotly_chart(fig2, key="th", on_select="rerun")

st.markdown("We can clearly see that including the interactive feature in the feature space immediately"
            "helps the model predict rainfall as it shows trend where the median value shows an increase in rainfall.  "
            "Perhaps this is indicative of some environmental conditions which could cause rainfall to increase.")

st.subheader("Apply your own transform!", divider="rainbow")
st.markdown("1. Choose a feature to transform!")
feat = st.selectbox("Features",
             ("Dewpoint", "Temperature", "Humidity (relative)", "Cloudiness"),
             index=None)

apply_to = np.abs(f_e_sub["dewpoint (degC)"])
if feat == "Dewpoint":
    apply_to = np.abs(f_e_sub["dewpoint (degC)"])
    st.markdown(f"**What is {feat}?**")
    st.markdown("The dewpoint is the temperature below which water can start to condense, turn into droplets and form dew."
                "It varies with pressure and temperature.  If the air temperature is the same as the dewpoint, then clouds and fog form"
                "which is why it can be useful in predicting rainfall")
if feat == "Temperature":
    apply_to = f_e_sub["Mean Temperature [Deg C]"]
    st.markdown(f"**What is {feat}?**")
    st.markdown("A lower temperature means that water clouds would be more likely to condense and precipitate, which is why it is a "
                "strong predictor of rainfall.")
if feat == "Humidity (relative)":
    apply_to = f_e_sub["Mean Relative Humidity [percent]"]
    st.markdown(f"**What is {feat}?**")
    st.markdown("The relative humidity is how much water vapour is in the air.  Humidity is proportional to rainfall, and is generally a strong "
                "indicator and predictor of the rain.")
if feat == "Cloudiness":
    apply_to = f_e_sub["cloudiness"]
    st.markdown(f"**What is {feat}?**")
    st.markdown("Expressed in hours, the cloudiness tells us how much of the sky was covered during a given hour.  The longer the sky was covered by clouds"
                ", the more likely it would be for rain to fall.")

st.markdown("2. Choose transformation to apply!")
trans = st.selectbox("Transformations",
             ("Log Transform", "Standard Scaling", "Rolling Average", "Lag"),
             index=None)

transformed = apply_to
if trans == "Log Transform":
    st.markdown(f"**What does {trans} do?**")
    st.markdown("The log transform applies the ln(x) function to all of the values in a given feature.  It helps"
                " in bringing values which are skewed onto a more normal distribution.  Great "
                "for polynomial regression (with regularisation applied) and SVR models. ")
    transformed = np.log(apply_to)

if trans == "Standard Scaling":
    st.markdown(f"**What does {trans} do?**")
    st.markdown("Standard scaling brings values of features which span orders of magnitude onto the same scale by using the z-score. "
                " In essence, our values will have a mean of 0 and a standard deviation of 1. "
                " Helps with polynomial regression and KNeighbours models mostly, but isn't as effective with tree-based models "
                "such as RandomForest or XGBoost.")
    stdscaler = StandardScaler()
    transformed = pd.Series(stdscaler.fit_transform(apply_to.values.reshape(-1,1)).ravel(),index=apply_to.index)

if trans == "Rolling Average":
    st.markdown(f"**What does {trans} do?**")
    st.markdown("A rolling average can help show periodical trends in features.  For example, a six hour rolling"
                "average for pressure can show the six hourly trend in pressure. Really helps tree-based models.")
    avg = st.slider("Select rolling average window(hrs)", 1, 1, 12,key="rolldemo")
    transformed = apply_to.rolling(window=avg,min_periods=1).mean().shift(1)

if trans == "Lag":
    st.markdown(f"**What does {trans} do?**")
    st.markdown("A lag offsets values by a certain period, and then reveals them to the model.  Essentially, you show the model"
                "past values, which it can then use to try predict future values.  Really helpful in predicting temporal-based data"
                " and boosts several models' performances. ")
    lag = st.slider("Select lag duration (hrs)", 1,1,12, key="lagdemo")
    transformed = apply_to.shift(lag).ffill()

st.markdown("3. Apply Transformation and see results!")
res = pd.DataFrame({"Before": apply_to,
                    "After": transformed}, index=f_e_sub.index)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

with st.expander("APPLY!"):
    st.dataframe(res)
st.markdown("**We can also observe the affect on a small scale ML model (takes a bit of time, the model is actually running in the background)**")
f_e_sub_ut = f_e_sub.drop("Unnamed: 0", axis=1).copy()
f_e_sub_ut["user feature"] = transformed.fillna(0)
nt_scores = []
t_scores = []
for regressors in [KNeighborsRegressor(), Ridge(), RandomForestRegressor()]:
    nt_scores.append(-cross_val_score(regressors, f_e_sub.drop("Rainfall [mm]", axis=1),
                                     f_e_sub["Rainfall [mm]"], scoring='neg_mean_absolute_error',cv=5,n_jobs=-1).mean())

for regressors in [KNeighborsRegressor(), Ridge(), RandomForestRegressor()]:
    t_scores.append(-cross_val_score(regressors, f_e_sub_ut.drop("Rainfall [mm]", axis=1),
                                     f_e_sub_ut["Rainfall [mm]"], scoring='neg_mean_absolute_error',cv=5,n_jobs=-1).mean())

r2 = pd.DataFrame({"MAE Without Transformation": nt_scores,
              "MAE With Transformation": t_scores}, index=["KNeighbours", "Ridge", "RandomForest"])

st.markdown("Comparing the mean absolute error (MAE) for separate models:")
st.dataframe(r2)
st.markdown("The mean MAE is a kind performance metric which show us the error in our predictions.  The MAE"
            " is the how off we are on average, and we generally want to try minimise this value as much as possible."
            " From the results you can see that some transforms are more appropriate for some models compared to others.")
