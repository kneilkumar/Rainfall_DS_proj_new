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

reg_imp_man = pd.read_csv("pages/pages/feat_imp_reg_for_dash.csv")
class_imp_man = pd.read_csv("pages/pages/feat_imp_class_for_dash.csv")
reg_imp_alb = pd.read_csv("pages/pages/regimp_alb.csv")
class_imp_alb = pd.read_csv("pages/pages/classimp_alb.csv")
reg_imp_heads = pd.read_csv("pages/pages/regimp_heads.csv")
class_imp_heads = pd.read_csv("pages/pages/classimp_heads.csv")
class_imp_motat = pd.read_csv("pages/pages/classimp_motat.csv")
reg_imp_motat = pd.read_csv("pages/pages/regimp_motat.csv")

st.header("Model Performance and Metrics", divider="rainbow")

with st.expander("See Metrics and Performance explanation"):
    st.markdown("The mean absolute error (MAE) is a common metric used in quantifying the error in rainfall models."
                " Comparing it against the average rainfall is a good method of benchmarking.  This is because hourly rainfall"
                " is zero most of the time, so guessing the average actually beats out most models right off the bat.")
    st.markdown("For the same reason, it is easier to guess that it won't rain, which is why the classifier baseline is 0."
                "  The f1 score is the proportion of how many correct guesses the classifier made.")
    st.markdown("The main metric(s) are the out of sample data related metrics (mean and MAE), since it truly tests the model's ability to "
                " predict data it has never seen before.  ")


mangere, albany, heads, motat = st.tabs(["Mangere", "Albany", "Manukau Heads", "MOTAT"])

with mangere:
    st.subheader("Regression Model (XGBoostRegressor)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rainfall (pure rain)", "1.24 (mm/hr)")
    c3.metric("10-fold CV average MAE:", "0.677 (mm/hr)")
    st.subheader("Classification Model (XGBoostClassifier)")
    co1, co2, co3 = st.columns(3)
    co1.metric("Baseline f1 score", "0")
    co3.metric("10-fold CV average f1 score", "0.645")
    st.subheader("Out of Sample Model Performance")
    c01, c02, c03 = st.columns(3)
    c01.metric("Average Rainfall (out of sample data)", "0.13 (mm/hr)")
    c03.metric("Out of Sample MAE", "0.09 (mm/hr)")
    st.subheader("Top 10 Most Important Parameters for Mangere Data")
    st.markdown("**Regressor's most important features**")
    st.bar_chart(data=reg_imp_man.iloc[0:10].sort_values(by="feature_importance"), x="feature_name", y="feature_importance")
    st.markdown("**Classifier's most important features**")
    st.bar_chart(data=class_imp_man.iloc[0:10].sort_values(by="feature_importance"),x="feature_name", y="feature_importance")

with albany:
    st.subheader("Regression Model (XGBoostRegressor)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rainfall (pure rain)", "1.27 (mm/hr)")
    c3.metric("10-fold CV average MAE:", "0.775 (mm/hr)")
    st.subheader("Classification Model (XGBoostClassifier)")
    co1, co2, co3 = st.columns(3)
    co1.metric("Baseline f1 score", "0")
    co3.metric("10-fold CV average f1 score", "0.57")
    st.subheader("Out of Sample Model Performance")
    c01, c02, c03 = st.columns(3)
    c01.metric("Average Rainfall (out of sample data)", "0.142(mm/hr)")
    c03.metric("Out of Sample MAE", "0.123(mm/hr)")
    st.subheader("Top 10 Most Important Parameters for Albany Data")
    st.markdown("**Regressor's most important features**")
    st.bar_chart(data=reg_imp_alb.iloc[0:10].sort_values(by="feature_importance"), x="feature_name", y="feature_importance")
    st.markdown("**Classifier's most important features**")
    st.bar_chart(data=class_imp_alb.iloc[0:10].sort_values(by="feature_importance"),x="feature_name", y="feature_importance")

with heads:
    st.subheader("Regression Model (XGBoostRegressor)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rainfall (pure rain)", "1.379 (mm/hr)")
    c3.metric("10-fold CV average MAE:", "0.857 (mm/hr)")
    st.subheader("Classification Model (XGBoostClassifier)")
    co1, co2, co3 = st.columns(3)
    co1.metric("Baseline f1 score", "0")
    co3.metric("10-fold CV average f1 score", "0.619")
    st.subheader("Out of Sample Model Performance")
    c01, c02, c03 = st.columns(3)
    c01.metric("Average Rainfall (out of sample data)", "0.188(mm/hr)")
    c03.metric("Out of Sample MAE", "0.150(mm/hr)")
    st.subheader("Top 10 Most Important Parameters for Manukau Heads Data")
    st.markdown("**Regressor's most important features**")
    st.bar_chart(data=reg_imp_heads.iloc[0:10].sort_values(by="feature_importance"), x="feature_name",
                 y="feature_importance")
    st.markdown("**Classifier's most important features**")
    st.bar_chart(data=class_imp_heads.iloc[0:10].sort_values(by="feature_importance"), x="feature_name",
                 y="feature_importance")

with motat:
    st.subheader("Regression Model (XGBoostRegressor)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rainfall (pure rain)", "1.257 (mm/hr)")
    c3.metric("10-fold CV average MAE:", "0.650 (mm/hr)")
    st.subheader("Classification Model (XGBoostClassifier)")
    co1, co2, co3 = st.columns(3)
    co1.metric("Baseline f1 score", "0")
    co3.metric("10-fold CV average f1 score", "0.657")
    st.subheader("Out of Sample Model Performance")
    c01, c02, c03 = st.columns(3)
    c01.metric("Average Rainfall (out of sample data)", "0.140 (mm/hr)")
    c03.metric("Out of Sample MAE", "0.090 (mm/hr)")
    st.subheader("Top 10 Most Important Parameters for MOTAT Data")
    st.markdown("**Regressor's most important features**")
    st.bar_chart(data=reg_imp_motat.iloc[0:10].sort_values(by="feature_importance"), x="feature_name",
                 y="feature_importance")
    st.markdown("**Classifier's most important features**")
    st.bar_chart(data=class_imp_motat.iloc[0:10].sort_values(by="feature_importance"), x="feature_name",
                 y="feature_importance")


