import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("Air Quality Prediction (PM2.5)")
st.write("Enter the values for each pollutant and time-based feature to predict PM2.5.")

# Pollutant input fields
co = st.number_input('CO (Carbon Monoxide)', value=0.5)
no = st.number_input('NO (Nitric Oxide)', value=10.0)
no2 = st.number_input('NO2 (Nitrogen Dioxide)', value=20.0)
o3 = st.number_input('O3 (Ozone)', value=30.0)
so2 = st.number_input('SO2 (Sulfur Dioxide)', value=5.0)
pm10 = st.number_input('PM10', value=60.0)
nh3 = st.number_input('NH3', value=15.0)

# Date/Time features
month = st.number_input('Month', min_value=1, max_value=12, value=5)
hour = st.number_input('Hour (0-23)', min_value=0, max_value=23, value=12)
year = st.number_input('Year', min_value=2014, max_value=2025, value=2025)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
day_of_week = st.number_input('Day of Week (0=Mon, 6=Sun)', min_value=0, max_value=6, value=2)
is_weekend = st.selectbox('Is it weekend?', [0, 1])

# Prepare input
input_data = np.array([[co, no, no2, o3, so2, pm10, nh3, month, hour, year, day, day_of_week, is_weekend]])

# Predict
if st.button("Predict PM2.5"):
    prediction = model.predict(input_data)
    st.success(f"Predicted PM2.5 Level: {prediction[0]:.2f} µg/m³")

st.title("Data Visualization")

import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
daily_data = pd.read_excel("D:\Excel sheets\Air Quality Data.xlsx")

subset = daily_data[['pm2_5', 'pm10', 'no2(Nitrogen Dioxide)', 'so2(Sulfur Dioxide)',
                     'co(Carbon Monoxide)', 'o3(Ozone)', 'nh3']]
fig = sns.pairplot(subset)
fig.fig.suptitle("Pairwise Relationships between Pollutants", y=1.02)
st.pyplot(fig)

fig = px.line(daily_data, x='date', y='pm2_5', title='PM2.5 Levels in Delhi Over Time')
fig.update_layout(xaxis_title='Date', yaxis_title='PM2.5')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='pm10', title='PM10 Levels in Delhi Over Time')
fig.update_traces(line=dict(color='red'))
fig.update_layout(xaxis_title='Date', yaxis_title='PM10')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='no2(Nitrogen Dioxide)', title='Nitrogen Dioxide Levels in Delhi Over Time')
fig.update_traces(line=dict(color='green'))
fig.update_layout(xaxis_title='Date', yaxis_title='Nitrogen Dioxide')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='so2(Sulfur Dioxide)', title='Sulfur Dioxide Levels in Delhi Over Time')
fig.update_traces(line=dict(color='orange'))
fig.update_layout(xaxis_title='Date', yaxis_title='Sulfur Dioxide')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='o3(Ozone)', title='Ozone Levels in Delhi Over Time')
fig.update_traces(line=dict(color='blue'))
fig.update_layout(xaxis_title='Date', yaxis_title='Ozone')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='co(Carbon Monoxide)', title='Carbon Monoxide Levels in Delhi Over Time')
fig.update_traces(line=dict(color='purple'))
fig.update_layout(xaxis_title='Date', yaxis_title='Carbon Monoxide')
st.plotly_chart(fig)

fig = px.line(daily_data, x='date', y='nh3', title='NH3Levels in Delhi Over Time')
fig.update_traces(line=dict(color='aqua'))
fig.update_layout(xaxis_title='Date', yaxis_title='NH3')
st.plotly_chart(fig)

fig = px.line(daily_data,
              x='date',
              y=['pm2_5', 'pm10', 'no2(Nitrogen Dioxide)', 'so2(Sulfur Dioxide)',
                 'co(Carbon Monoxide)', 'o3(Ozone)', 'nh3'],
              title='Air Pollutant Levels in Delhi Over Time')
fig.update_layout(xaxis_title='Date', yaxis_title='Pollutant Levels')
st.plotly_chart(fig)

fig = px.scatter(daily_data, x='no2(Nitrogen Dioxide)', y='pm2_5',
                 title='NO2 vs PM2.5', trendline='ols', color='pm2_5')
fig.update_layout(xaxis_title='NO2', yaxis_title='PM2.5')
st.plotly_chart(fig)

avg_pollutants = daily_data[['pm2_5', 'pm10', 'no2(Nitrogen Dioxide)', 'so2(Sulfur Dioxide)',
                             'co(Carbon Monoxide)', 'o3(Ozone)', 'nh3']].mean().reset_index()
avg_pollutants.columns = ['Pollutant', 'Average_Level']

fig = px.bar(avg_pollutants, x='Pollutant', y='Average_Level',
             title='Average Levels of Pollutants in Delhi', color='Pollutant')
st.plotly_chart(fig)

plt.figure(figsize=(10, 8))
sns.heatmap(daily_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Air Quality Features")
st.pyplot(plt.gcf())

plt.figure(figsize=(8, 5))
sns.histplot(daily_data['pm2_5'], bins=30, kde=False, color='orange')
plt.title('PM2.5 Distribution')
plt.xlabel('PM2.5')
plt.ylabel('Count')
st.pyplot(plt.gcf())



fig = px.box(daily_data, y='pm2_5', title='Boxplot of PM2.5 Levels')

st.plotly_chart(fig)


