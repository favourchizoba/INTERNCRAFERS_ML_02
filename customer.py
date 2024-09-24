

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Copy the data
ds = data.copy()

# Rename the columns
ds.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)

# Encode the 'Gender' column
label_encoder = LabelEncoder()
ds['Gender'] = label_encoder.fit_transform(ds['Gender'])

# Select relevant features for clustering
features = ds[['Gender', 'Age', 'Income', 'Score']]

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Preprocess the data for clustering
x = ds[['Income', 'Score']]

# KMeans Clustering
kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=100, random_state=42)
cluster_predict = kmeans_model.fit_predict(x)
x['Cluster'] = cluster_predict

# # Visualization using Plotly
dfs = x.astype({"Cluster": "object"})
dfs = dfs.sort_values("Cluster")

fig = px.scatter(
    dfs,
    x='Score',
    y='Income',
    color='Cluster',
    title='Customer Clustering Segments based on Annual Income and Spending Score'
)


# Streamlit app layout
st.title('Customer Segmentation using K-means Clustering')

# Add an image (optional)
st.image('pngwing.com (31).png', width=300)

# # Sidebar image
# st.sidebar.image('pngwing.com (32).png', width=200, caption='Welcome User')
# st.sidebar.markdown('---')

# Display project background information
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Project Background Information</h2>", 
    unsafe_allow_html=True
)
st.write("This project uses K-means clustering to segment customers based on their purchase history, allowing the store to better understand its customer base.")

# Display dataset overview
if st.checkbox('Show Raw Dataset'):
    st.write(data)

# Display clustering results
if st.checkbox('Show Clustering Visuals '):
    st.plotly_chart(fig)

# Map cluster interpretations
x['ClusterInterprete'] = x.Cluster.map({
 3:'HighIncome_LowSpend',
 1: 'HighIncome_HighSpend',
 2: 'LowIncome_HighSpend',
 4: 'LowIncome_LowSpend',
 0: 'MidInc_MidSpend'
})

# Display clustered dataset
if st.checkbox("Show Clustered Dataset"):
    st.write(x)

# Plotting the bar chart
if st.checkbox("Show Bar Chart of Customer Segments"):
    plt.figure(figsize=(12, 4))
    fig = sns.barplot(x=x.ClusterInterprete, y=x.Income, palette='Set2', ci=0)
    for i in fig.containers:
        fig.bar_label(i)
    plt.title('Customer Segments based on Annual Income and Spending Score')
    st.pyplot(plt)

# User Guide and Help Section
st.header('User Guide & Help')

if st.checkbox('Show User Guide'):
    st.subheader('User Guide')
    st.write("""
    - Use the number inputs to provide customer information.
    - Click to check the clustering visualization
    - Click 'Show Clustered Dataset' to gain insights on K-means clustering.
    - Check 'Show Raw Dataset' to explore the customer data.
    - Check for Cluster Interpretation to understand customer segments.
    """)

    st.subheader('Need Help?')
    st.write("""
    - If you encounter any issues or have questions, please contact our support team at chibuezechizobafavour@gmail """)




















