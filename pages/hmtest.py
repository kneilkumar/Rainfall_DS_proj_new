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
from streamlit_folium import st_folium, folium_static

mangere_df = pd.read_csv("pages/csv_mangere_for_dash").sort_values(by="Observation time UTC")
albany_df = pd.read_csv("pages/csv_albany_for_dash").sort_values(by="Observation time UTC")
heads_df = pd.read_csv("pages/csv_manheads_for_dash").sort_values(by="Observation time UTC")
motat_df = pd.read_csv("pages/csv_motat_for_dash").sort_values(by="Observation time UTC")

akl_map = folium.Map(location=(-36.8059, 174.7645))

man_2023_rain = list(mangere_df["Rainfall [mm]"].iloc[6058:7799])
heads_2023_rain = list(heads_df["Rainfall [mm]"].iloc[0:1741])
albany_2023_rain = list(albany_df["Rainfall [mm]"].iloc[21190:22931])
motat_2023_rain = list(motat_df["Rainfall [mm]"].iloc[10830:12571])

albany_lat = [-36.7269 for i in range(1741)]
albany_long = [174.698 for j in range(1741)]
motat_lat = [-36.8678 + np.random.uniform(-0.01, 0.01) for i in range(1741)]
motat_long = [174.7269 + np.random.uniform(-0.01, 0.01) for i in range(1741)]
heads_lat = [-37.0509 + np.random.uniform(-0.01, 0.01) for i in range(1741)]
heads_long = [174.6113 + np.random.uniform(-0.01, 0.01) for i in range(1741)]
man_lat = [-36.9722 + np.random.uniform(-0.01, 0.01) for i in range(1741)]
man_long = [174.7867 + np.random.uniform(-0.01, 0.01) for i in range(1741)]

albany_pt = [[albany_lat[i], albany_long[i], albany_2023_rain[i]] for i in range(len(albany_2023_rain))]
man_pt = [[man_lat[i], man_long[i], man_2023_rain[i]] for i in range(len(man_2023_rain))]
heads_pt = [[heads_lat[i],heads_long[i], heads_2023_rain[i]] for i in range(len(heads_2023_rain))]
motat_pt = [[motat_lat[i], motat_long[i], motat_2023_rain[i]] for i in range(len(motat_2023_rain))]

albany_data = []
for i in range(len(albany_2023_rain)):
    temp = []
    for j in range(50):
        temp.append([-36.7269 + np.random.uniform(-0.05, 0.05),   # jitter here
            174.698 + np.random.uniform(-0.05, 0.05),
            albany_2023_rain[i]])
    albany_data.append(temp)

mangere_data = []
for i in range(len(man_2023_rain)):
    temp = []
    for j in range(50):
        temp.append([-36.9722 + np.random.uniform(-0.05, 0.05),   # jitter here
            174.7867 + np.random.uniform(-0.05, 0.05),
            man_2023_rain[i]])
    mangere_data.append(temp)

heads_data = []
for i in range(len(heads_2023_rain)):
    temp = []
    for j in range(50):
        temp.append([-37.0509 + np.random.uniform(-0.05, 0.05),   # jitter here
            174.6113 + np.random.uniform(-0.05, 0.05),
            man_2023_rain[i]])
    heads_data.append(temp)

motat_data = []
for i in range(len(motat_2023_rain)):
    temp = []
    for j in range(50):
        temp.append([-36.8678 + np.random.uniform(-0.05, 0.05),   # jitter here
            174.7269 + np.random.uniform(-0.05, 0.05),
            man_2023_rain[i]])
    motat_data.append(temp)

start = datetime.datetime(2022, 4,4,0)
end = datetime.datetime(2023, 4,4, 0)
times = [str(start + datetime.timedelta(hours=i)) for i in range(1741)]

HeatMapWithTime(albany_data, index=times, radius=50, show=True, control=True,overlay=True).add_to(akl_map)
HeatMapWithTime(motat_data,radius=50, overlay=True, index=times).add_to(akl_map)
HeatMapWithTime(heads_data,radius=50, overlay=True, index=times).add_to(akl_map)
HeatMapWithTime(mangere_data,radius=50, overlay=True, index=times).add_to(akl_map)
folium.LayerControl().add_to(akl_map)

akl_map.save("auckland_rain_map.html")

mangere_tab, heads_tab, motat_tab, albany_tab = st.tabs(["Mangere", "Manukau Heads", "Motat", "Albany"])
with mangere_tab:
    leftman, rightman = st.columns(2)
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    if leftman.checkbox("Show Predicted Rainfall", width="stretch"):
        fig.add_trace(
            go.Scatter(x=mangere_df["Observation time UTC"],
                       y=mangere_df["predicted rainfall [mm]"],
                       name="predicted rainfall"),
            secondary_y=False
        )

    if rightman.checkbox("Show True Rainfall", width="stretch"):
        fig.add_trace(
            go.Scatter(x=mangere_df["Observation time UTC"],
                       y=mangere_df["True Rainfall [mm]"],
                       name="true rainfall"),
            secondary_y=True
        )

    fig.update_layout(title_text="Predicted Rainfall vs True Rainfall (Mangere)")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Rainfall [mm/hr]", secondary_y=True)
    st.plotly_chart(fig, config={'scrollZoom': False})

    roll_show = st.toggle("Show Rolling Average Graph", key="man")
    if roll_show:
        rollav_mangere = st.slider("Select rolling average window", 7, 25, 360)
        mangere_df[f"Rolling Average: {rollav_mangere} Day Window"] = mangere_df['Rainfall [mm]'].rolling(
            window=rollav_mangere).mean()

        fig05 = make_subplots()
        fig05.add_trace(
            go.Scatter(x=mangere_df["Observation time UTC"],
                       y=mangere_df[f"Rolling Average: {rollav_mangere} Day Window"],
                       name="Rainfall Rolling Average [mm/hr]")
        )
        fig05.update_layout(title_text="Rolling Average Rainfall vs Time (Mangere)")
        fig05.update_xaxes(title_text="Time")
        fig05.update_yaxes(title_text=f"Rainfall ({rollav_mangere} day rolling average) [mm/hr]")
        st.plotly_chart(fig05, config={'scrollZoom': False})

with heads_tab:
    leftheads, rightheads = st.columns(2)
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])

    if leftheads.checkbox("Show Predicted Rainfall", width="stretch", key="head1"):
        fig1.add_trace(
            go.Scatter(x=heads_df["Observation time UTC"],
                       y=heads_df["predicted rainfall [mm]"],
                       name="predicted rainfall"),
            secondary_y=False
        )

    if rightheads.checkbox("Show True Rainfall", width="stretch", key="head2"):
        fig1.add_trace(
            go.Scatter(x=heads_df["Observation time UTC"],
                       y=heads_df["True Rainfall [mm]"],
                       name="true rainfall"),
            secondary_y=True
        )

    fig1.update_layout(title_text="Predicted Rainfall vs True Rainfall (Manukau Heads)")
    fig1.update_xaxes(title_text="Time")
    fig1.update_yaxes(title_text="Rainfall [mm/hr]", secondary_y=True)
    st.plotly_chart(fig1, config={'scrollZoom': False})

    roll_show_heads = st.toggle("Show Rolling Average Graph", key="mheads")
    if roll_show_heads:
        rollav_heads = st.slider("Select rolling average window", 7, 25, 360, key="head")
        heads_df[f"Rolling Average: {rollav_heads} Day Window"] = heads_df['Rainfall [mm]'].rolling(
            window=rollav_heads).mean()

        fig25 = make_subplots()
        fig25.add_trace(
            go.Scatter(x=heads_df["Observation time UTC"],
                       y=heads_df[f"Rolling Average: {rollav_heads} Day Window"],
                       name="Rainfall Rolling Average [mm/hr]")
        )
        fig25.update_layout(title_text="Rolling Average Rainfall vs Time (Manukau Heads)")
        fig25.update_xaxes(title_text="Time")
        fig25.update_yaxes(title_text=f"Rainfall ({rollav_heads} day rolling average) [mm/hr]")
        st.plotly_chart(fig25, config={'scrollZoom': False})

with motat_tab:
    lefttat, righttat = st.columns(2)
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    if lefttat.checkbox("Show Predicted Rainfall", width="stretch", key="tat1"):
        fig2.add_trace(
            go.Scatter(x=motat_df["Observation time UTC"],
                       y=motat_df["predicted rainfall [mm]"],
                       name="predicted rainfall"),
            secondary_y=False
        )

    if righttat.checkbox("Show True Rainfall", width="stretch", key="tat2"):
        fig2.add_trace(
            go.Scatter(x=motat_df["Observation time UTC"],
                       y=motat_df["True Rainfall [mm]"],
                       name="true rainfall"),
            secondary_y=True
        )

    fig2.update_layout(title_text="Predicted Rainfall vs True Rainfall (Motat)")
    fig2.update_xaxes(title_text="Time")
    fig2.update_yaxes(title_text="Rainfall [mm/hr]", secondary_y=True)
    st.plotly_chart(fig2, config={'scrollZoom': False})

    roll_show_tat = st.toggle("Show Rolling Average Graph", key="mtat")
    if roll_show_tat:
        rollav_motat = st.slider("Select rolling average window", 7, 25, 360, key="tat")
        motat_df[f"Rolling Average: {rollav_motat} Day Window"] = motat_df['Rainfall [mm]'].rolling(
            window=rollav_motat).mean()

        fig25 = make_subplots()
        fig25.add_trace(
            go.Scatter(x=motat_df["Observation time UTC"],
                       y=motat_df[f"Rolling Average: {rollav_motat} Day Window"],
                       name="Rainfall Rolling Average [mm/hr]")
        )
        fig25.update_layout(title_text="Rolling Average Rainfall vs Time (Motat)")
        fig25.update_xaxes(title_text="Time")
        fig25.update_yaxes(title_text=f"Rainfall ({rollav_motat} day rolling average) [mm/hr]")
        st.plotly_chart(fig25, config={'scrollZoom': False})

with albany_tab:
    leftban, rightban = st.columns(2)
    fig3 = make_subplots(specs=[[{'secondary_y': True}]])

    if leftban.checkbox("Show Predicted Rainfall", width="stretch", key="ban1"):
        fig3.add_trace(
            go.Scatter(x=albany_df["Observation time UTC"],
                       y=albany_df["predicted rainfall [mm]"],
                       name="predicted rainfall"),
            secondary_y=False
        )

    if rightban.checkbox("Show True Rainfall", width="stretch", key="ban2"):
        fig3.add_trace(
            go.Scatter(x=albany_df["Observation time UTC"],
                       y=albany_df["True Rainfall [mm]"],
                       name="true rainfall"),
            secondary_y=True
        )

    fig3.update_layout(title_text="Predicted Rainfall vs True Rainfall (Albany)")
    fig3.update_xaxes(title_text="Time")
    fig3.update_yaxes(title_text="Rainfall [mm/hr]", secondary_y=True)
    st.plotly_chart(fig3, config={'scrollZoom': False})

    roll_show_ban = st.toggle("Show Rolling Average Graph", key="bany")
    if roll_show_ban:
        rollav_albany = st.slider("Select rolling average window", 7, 25, 360, key="ban")
        albany_df[f"Rolling Average: {rollav_albany} Day Window"] = albany_df['Rainfall [mm]'].rolling(
            window=rollav_albany).mean()

        fig35 = make_subplots()
        fig35.add_trace(
            go.Scatter(x=albany_df["Observation time UTC"],
                       y=albany_df[f"Rolling Average: {rollav_albany} Day Window"],
                       name="Rainfall Rolling Average [mm/hr]")
        )
        fig35.update_layout(title_text="Rolling Average Rainfall vs Time (Albany)")
        fig35.update_xaxes(title_text="Time")
        fig35.update_yaxes(title_text=f"Rainfall ({rollav_albany} day rolling average) [mm/hr]")
        st.plotly_chart(fig35, config={'scrollZoom': False})

# ---------- Spatial Maps ---------------#

st.header("Rainfall Across All Four Stations", divider="rainbow")

with st.expander("How to view data"):
    st.write('''guide goes here''')

st.link_button(label="Click here to view (file name: auckland_rain_map.html)",
               url="https://github.com/kneilkumar/Rainfall_DS_proj_new.git")

with st.expander("Setup Explanation"):
    st.write('''explanation goes here''')

# ------------------- DataWorkflow demo --------------- #


