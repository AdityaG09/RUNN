import streamlit as st
import pandas as pd
import numpy as np

st.title('RUNN')

# This is used to create a select box of unique ailments
data2 = pd.read_csv("./merged2.csv")
options = data2.REASONDESCRIPTION.unique()

disease = st.selectbox("Select a disease you want to find", options)
st.write("You are searching for ", disease)

@st.cache
def load_data(nrows):
    data = pd.read_csv("./merged7.csv")
    return data

data_load_state = st.text('Loading data...')
data = load_data(100)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.subheader('Hospital Density Map')
filtered_data = (data[data.REASONDESCRIPTION == disease])

x = filtered_data.LON
y = filtered_data.LAT

#GAUSS MAP
from scipy.stats.kde import gaussian_kde
k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7,8))
ax2 = fig.add_subplot(212)

# alpha=0.5 will make the plots semitransparent
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(y.min(), y.max())

im = plt.imread('./map.png')
ax2.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
st.pyplot(fig)


# 3D MAP
df = data
#df['UTILIZATION']= df['UTILIZATION'].astype(float)

import pydeck as pdk

view = pdk.data_utils.compute_view(df[["LON", "LAT"]])
view.pitch = 50
view.bearing = 0

column_layer = pdk.Layer(
            "ColumnLayer",
            data=df,
            get_position=['LON', 'LAT'],
            get_elevation='UTILIZATION',
            radius=500,
            elevation_scale=10,
            
            get_fill_color=["UTILIZATION * 10", "UTILIZATION", "UTILIZATION * 10", 140],
            pickable=True,
            extruded=True,
            auto_highlight=True
         )

tooltip = {
    "html": "<b>{NAME}</b> <BR> <b>{ADDRESS}</b> <BR> <b> Patients admitted = {UTILIZATION}",
    "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
}


r = pdk.Deck(
    column_layer,
    initial_view_state=view,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v9",
)

st.pydeck_chart(r)