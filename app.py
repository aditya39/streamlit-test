import streamlit as st
import pandas as pd
import pycaret.classification as pc
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title('Data Science Project App')
st.write('This app allows you to load data, preprocess/clean it, train a machine learning model, visualize data, and make inferences with new data.')

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Load Data", "Data Preprocessing/Cleaning", "Machine Learning Training", "Visualize Data", "Inference"]
section = st.sidebar.radio("Go to", sections)

# Load Data
if section == "Load Data":
    st.header("Load Data")
    data_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.session_state['data'] = data
        st.write("Data Loaded Successfully")
        st.write(data.head())

# Data Preprocessing/Cleaning
elif section == "Data Preprocessing/Cleaning":
    st.header("Data Preprocessing/Cleaning")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Original Data")
        st.write(data.head())
        
        # Example preprocessing: Drop missing values
        processed_data = data.dropna()
        st.session_state['processed_data'] = processed_data
        st.write("Processed Data (missing values dropped)")
        st.write(processed_data.head())
    else:
        st.write("Please load the data first")

# Machine Learning Training
elif section == "Machine Learning Training":
    st.header("Machine Learning Training")
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        st.write("Setting up PyCaret...")
        
        # Setup the PyCaret environment and train model
        setup = pc.setup(data, target='target', silent=True, html=False)  # Adjust 'target' as needed
        best_model = pc.compare_models()
        
        st.session_state['best_model'] = best_model
        st.write("Best Model Trained:", best_model)
    else:
        st.write("Please preprocess the data first")

# Visualize Data
elif section == "Visualize Data":
    st.header("Visualize Data")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Data Visualization")
        visualization_type = st.selectbox("Choose a visualization", ["Histogram", "Scatter Plot", "Correlation Heatmap"])
        
        if visualization_type == "Histogram":
            column = st.selectbox("Select Column", data.columns)
            plt.figure(figsize=(10, 4))
            sns.histplot(data[column])
            st.pyplot(plt)
        elif visualization_type == "Scatter Plot":
            col1 = st.selectbox("Select X Column", data.columns)
            col2 = st.selectbox("Select Y Column", data.columns)
            plt.figure(figsize=(10, 4))
            sns.scatterplot(x=data[col1], y=data[col2])
            st.pyplot(plt)
        elif visualization_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 4))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)
    else:
        st.write("Please load the data first")

# Inference
elif section == "Inference":
    st.header("Inference with New Data")
    if 'best_model' in st.session_state:
        new_data_file = st.file_uploader("Upload new data for inference", type=["csv"])
        if new_data_file is not None:
            new_data = pd.read_csv(new_data_file)
            predictions = pc.predict_model(st.session_state['best_model'], data=new_data)
            st.write("Predictions:")
            st.write(predictions)
    else:
        st.write("Please train the model first")

