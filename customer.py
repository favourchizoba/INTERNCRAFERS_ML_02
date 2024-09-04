

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Title and header
st.markdown("<h1 style='color: #0C359E; text-align: center; font-size: 60px; font-family: Helvetica'>Customer Segmentation using K-means Clustering</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #F11A7B; text-align: center; font-family: Helvetica '>Built By Franches</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Add an image
st.image('pngwing.com (31).png', width=400)

# Sidebar image
st.sidebar.image('pngwing.com (32).png', width=200, caption='Welcome User')

# Add divider and spacing
st.sidebar.divider()

# Project background information
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Project Background Information</h2>", 
    unsafe_allow_html=True
)
st.write("This project focuses on using a K-means clustering algorithm to segment customers of a retail store based on their purchase history. By grouping customers with similar buying behaviors, the store can better understand its customer base, enabling more personalized marketing, targeted promotions, and optimized product offerings. The insights gained from this clustering will help the store improve customer satisfaction and make informed business decisions, ultimately enhancing overall performance.")
st.markdown("<br>", unsafe_allow_html=True)

# Dataset overview
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Customer Data</h2>", 
    unsafe_allow_html=True
)

if st.checkbox('Show Raw data'):
    st.write(data)

# Input fields for user data
def user_input():
    CustomerID = st.sidebar.number_input('CustomerID', float(data['CustomerID'].min()), float(data['CustomerID'].max()))
    income = st.sidebar.number_input('Income', float(data['Annual Income (k$)'].min()), float(data['Annual Income (k$)'].max()))
    spending_score = st.sidebar.number_input('Score', float(data['Spending Score (1-100)'].min()), float(data['Spending Score (1-100)'].max()))
    return CustomerID, income, spending_score

# Get user input
customer_id, income, spending_score = user_input()

# Prepare data for K-means clustering
user_data = pd.DataFrame([[income, spending_score]], columns=['Income', 'Score'])

# K-means clustering
kmeans = KMeans(n_clusters=5)  # Specify the number of clusters
data['Cluster'] = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])


# Map cluster interpretations
data['ClusterInterprete'] = data['Cluster'].map({
    0: 'MidInc_MidSpend',
    1: 'HiInc_LowSpend',
    2: 'LwInc_LwSpend',
    3: 'LwInc_HiSpend',
    4: 'HiInc_HiSpend'
})

# Display clustered data
if st.checkbox("Show Clustered Data"):
    st.write(data)

# Plotting the bar chart
if st.checkbox("Show Bar Chart of Customer Segments"):
    plt.figure(figsize=(12, 4))
    fig = sns.barplot(x='ClusterInterprete', y='Annual Income (k$)', data=data, palette='Set2', ci=None)
    for container in fig.containers:
        fig.bar_label(container)
    plt.title('Customer Segments based on Annual Income and Spending Score')
    st.pyplot(plt)


# User Guide and Help Section
st.header('User Guide & Help')

if st.checkbox('Show User Guide'):
    st.subheader('User Guide')
    st.write("""
    - Use the number inputs to provide customers information
    - Click 'Clustered data to have insight on kmeans clustering
    - Check 'Show Raw Data'to explore the customer data
    - Check for ClusterInterprete       
    """)

    st.subheader('Need Help?')
    st.write("""
    - If you encounter any issues or have questions, please contact our support team at chibuezechizobafavour@gmail.com
    """)