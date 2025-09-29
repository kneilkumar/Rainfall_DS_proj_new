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


# ----------- Station-Wide-Overview ------------- #
st.set_page_config(
    page_title="Hello!"
)
st.write("# Rainfall Prediction!")
st.sidebar.success("Navigate Dashboard")
st.markdown('''
This is a dashboard which walks you through a Machine Learning Rainfall Prediction Project.  
**Check the sidebar to see how I went and what I did!**  I hope you enjoy exploring this dashboard as much as I did making it 😊!
''')



