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
import seaborn as sns

st.header("System Overview")
st.subheader("Environmental Element Interaction", divider="rainbow")

# ------- upload dfs ------- #

fe_df = pd.read_csv("pages/pages/feat_eng_df.csv")
mangere = pd.read_csv("pages/pages/csv_mangere_for_dash")[["Sunshine [hrs]", "Rainfall [mm]", "Mean Temperature [Deg C]", "Mean Relative Humidity [percent]", "Direction [deg T]", "Speed [m/s]", "Mean sea level pressure [Hpa]", "dewpoint (degC)"]]
head = pd.read_csv("cpages/pages/sv_manheads_for_dash")[["Sunshine [hrs]", "Rainfall [mm]", "Mean Temperature [Deg C]", "Mean Relative Humidity [percent]", "Direction [deg T]", "Speed [m/s]", "Mean sea level pressure [Hpa]", "dewpoint (degC)"]]
motat = pd.read_csv("pages/pages/csv_motat_for_dash")[["Sunshine [hrs]", "Rainfall [mm]", "Mean Temperature [Deg C]", "Mean Relative Humidity [percent]", "Direction [deg T]", "Speed [m/s]", "Mean sea level pressure [Hpa]", "dewpoint (degC)"]]
albany = pd.read_csv("pages/pages/csv_albany_for_dash")[["Sunshine [hrs]", "Rainfall [mm]", "Mean Temperature [Deg C]", "Mean Relative Humidity [percent]", "Direction [deg T]", "Speed [m/s]", "Mean sea level pressure [Hpa]", "dewpoint (degC)"]]

st.markdown("**Its always a good idea to see how different elements of the system interact with each other.  "
            "This allows us to:**")
st.markdown("- Find correlations between features; leads to more feature engineering")
st.markdown("- Which features are most influential in which region")
st.markdown("- Boosts overall understanding of rainfall patterns in Auckland")

st.subheader("Feature Correlation: Matrix and Heatmap")

st.markdown("Pearson correlation coefficients are a good metric for evaluating how closely related two features are with each other")
st.markdown("- -1 means there's a negative linear relationship")
st.markdown("- +1 means there's a positivelinear relationship")
st.markdown("- 0 means there's a no linear relationship")
st.markdown("Values tend to lie in between -1 and 1")

station = st.selectbox("Station-specific correlations",
                       ("Mangere", "Manukau Heads", "Albany", "Motat"))

if station == "Mangere":
    fig, ax = plt.subplots()
    corr_mang = mangere.corr()
    sns.heatmap(corr_mang,ax=ax,cbar_kws={'label':'correlation strength'})
    st.write(fig)
    ax.set_title("Pearson Correlation Plot for Mangere")
    st.markdown("")
    st.markdown("**Correlation Table**")
    st.dataframe(corr_mang)

if station == "Manukau Heads":
    fig, ax = plt.subplots()
    corr_man = head.corr()
    sns.heatmap(corr_man,ax=ax,cbar_kws={'label':'correlation strength'})
    st.write(fig)
    ax.set_title("Pearson Correlation Plot for Manukau Heads")
    st.markdown("")
    st.markdown("**Correlation Table**")
    st.dataframe(corr_man)

if station == "Motat":
    fig, ax = plt.subplots()
    corr_mot = motat.corr()
    sns.heatmap(corr_mot,ax=ax, cbar_kws={'label':'correlation strength'})
    ax.set_title("Pearson Correlation Plot for Motat")
    st.write(fig)
    st.markdown("")
    st.markdown("**Correlation Table**")
    st.dataframe(corr_mot)

if station == "Albany":
    fig, ax = plt.subplots()
    corr_al = albany.corr()
    sns.heatmap(corr_al,ax=ax,cbar_kws={'label':'correlation strength'})
    ax.set_title("Pearson Correlation Plot for Albany")
    st.write(fig)
    st.markdown("")
    st.markdown("**Correlation Table**")
    st.dataframe(corr_al)

st.subheader("Key Takeaways:")
st.markdown('- Humidity and Temperature have a strong positive correlation, regardless of location.')
st.markdown('- Humidity and Sunshine have a strong negative correlation, regardless of location.')
st.markdown('- At Manukau Heads, the rainfall seems to share a semi-strong positive relationship with the humidity,'
            ' while having a semi-strong negative relationship to the Sea Level Pressure.  This is the only location where'
            ' this relationship is observed so vividly.')
st.markdown('- At Albany, a lot of the features seem to be barely related to each other in a linear manner.')
st.markdown('- Motat and Albany seem to have similar weather patterns, as their correlation maps are quite similar')
st.markdown('- Mangere has the most variation when it comes to correlative qualities across features, as there are strong negative and positive'
            'linear correlations across the board between features.')
