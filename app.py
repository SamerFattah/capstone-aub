import streamlit as st
from streamlit_option_menu import option_menu 


from PIL import Image

import json
import base64
import requests
import random
import cloudpickle

#data exploration
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
#ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error



# Function to download file from Google Drive
def download_file_from_google_drive(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    return response.content

@st.cache_data
def load_data():
    # Replace these URLs with your Google Drive direct download links
    # Replace these URLs with your Google Drive direct download links
    d2022_1 = 'https://drive.google.com/uc?export=download&id=13Owi2M5Lma--DYSJSVSwFrpZEWlMGM9q'
    d2022_2 = 'https://drive.google.com/uc?export=download&id=13HBDx4Zc4V9y9NzKZixxZO8juxcwBgXa'
    d2023_1 = 'https://drive.google.com/uc?export=download&id=13LGsfHoio4JeAi8iiIJfGterhM45Dce7'
    d2023_2 = 'https://drive.google.com/uc?export=download&id=13Jry7rWKNoymkHqxfdRmCvNt-cgjM3fT'

    # Download files
    data_2022_1 = download_file_from_google_drive(d2022_1)
    data_2022_2 = download_file_from_google_drive(d2022_2)
    data_2023_1 = download_file_from_google_drive(d2023_1)
    data_2023_2 = download_file_from_google_drive(d2023_2)

    # Read Excel files into pandas DataFrames
    df_2022_1 = pd.read_excel(pd.io.common.BytesIO(data_2022_1), engine='openpyxl')
    df_2022_2 = pd.read_excel(pd.io.common.BytesIO(data_2022_2), engine='openpyxl')
    df_2023_1 = pd.read_excel(pd.io.common.BytesIO(data_2023_1), engine='openpyxl')
    df_2023_2 = pd.read_excel(pd.io.common.BytesIO(data_2023_2), engine='openpyxl')
    # Creating a Function to clean the Data (Removing Irrelevent Columns)
    def remove_columns(df, columns):
      df = df.drop(columns, axis=1)
      return df
    # Concatenate DataFrames
    df_2022 = pd.concat([df_2022_1, df_2022_2])
    df_2023 = pd.concat([df_2023_1, df_2023_2])
    # Application of the above fuction [that will clean the data]
    columns_22 = ['Unnamed: 0','Project', 'SubTask', 'CraftCode','Unnamed: 20',
       'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24']
    #Application of the above fuction [that will clean the data]
    columns_23 = ['Unnamed: 0','Project', 'SubTask', 'CraftCode','Unnamed: 21',
       'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
       'Unnamed: 20']
    df_2022 = remove_columns(df_2022,columns_22)
    df_2023 = remove_columns(df_2023,columns_23)
    return df_2022, df_2023
df_2022, df_2023 = load_data()
#-----------------------------------------------------------------------------#
# Menu bar
# Create a horizontal option menu
tabs = option_menu(
    menu_title=None,  # No title for the menu
    options=['Company Overview', 'Data Exploration', 'Predictive Model'], 
    icons=['house', 'clipboard-data', 'chat-right-dots'], 
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "#EEF4F8",  # Overall menu background color
            "display": "flex",  # Use flexbox for alignment
            "justify-content": "center",  # Center the menu
        },
        "icon": {"color": "white", "font-size": "18px"},  # Icon style
        "nav": {
            "border": "none",  # Remove lines between menu items
            "background-color": "#EEF4F8",  # Ensure consistency in overall background
        },
        "nav-item": {"margin": "0px", "padding": "0px"},  # No extra spacing
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "padding": "8px",  # Adjust padding for better spacing
            "background-color": "#FFFFFF",  # Default menu background
            "border-radius": "4px",  # Optional rounded corners
            "color": "black",  # Default text color
        },
        "nav-link-selected": {
            "background-color": "#081D31",  # Keep selected tab white
            "color": "white",  # Text color for active tab
            "font-weight": "bold",  # Optional: Bold text for active tab
        },
    }
)

#--------------------------------------------------------------------------------------------------------------------------------------------#
# # Component1: Home
if tabs == "Company Overview":
    def company_overview_tab():
    # Custom CSS to override default padding and alignment
      st.markdown(
        """
        <style>
        /* Remove padding/margin from the main block container */
        .block-container {
            padding-left: 0rem;
            padding-right: 0rem;
        }
        /* Align the image to the left */
        .image-container {
            text-align: left;
            margin-top: 20px;
            margin-left: 0px; /* Ensure it starts from the true left */
        }
        /* Align the content to the left */
        .content-container {
            text-align: left;
            margin-top: 20px;
            margin-left: 0px; /* Ensure it starts from the true left */
            padding: 20px;  /* Optional: Add some padding for better readability */
            max-width: 100%;  /* Use full width for content */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Image Section
      st.markdown(
        """
        <div class="image-container">
        """,
        unsafe_allow_html=True,
    )
      st.image("assets/engineer.jpg")
      st.markdown("</div>", unsafe_allow_html=True)

    # Content Section
      st.markdown(
        """
        <div class="content-container">
        <p><br>Our Corporation is an American technology-focused engineering firm specializing in defense, intelligence, security, and critical infrastructure. The company provides advanced technology solutions primarily to government agencies and critical infrastructure clients.</p>

        <h2>Company Overview:</h2>
        <h2>What They Do:</h2>
        <p>We offer services in several key areas:</p>
        <ul>
          <li><strong>Defense and Intelligence:</strong> Cybersecurity solutions, missile defense technologies, space systems, and intelligence support.</li>
          <li><strong>Critical Infrastructure:</strong> Engineering and managing transportation projects like highways and transit systems, implementing smart city technologies, and providing environmental solutions.</li>
          <li><strong>Security and Technology:</strong> Protecting critical infrastructure assets, developing biometric identification systems, and offering data analytics, artificial intelligence, and cloud computing services.</li>
        </ul>

        <p>Our mission is to deliver innovative solutions that make the world safer, smarter, and more connected.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Call the function in the Company Overview tab
company_overview_tab()
if tabs == 'Data Exploration':
      # Data Preprocessing
  df_2023['Week End'] = pd.to_datetime(df_2023['Week End'], errors='coerce')
  df_2023['Total Hrs'] = df_2023['Reg Hrs'] + df_2023['OT Hrs']
  # Sidebar Filters
  st.sidebar.header("Filters")
  col11, col13, col12 = st.columns([2, 1, 1])  # Adjust column widths for better spacing
  with col11:
    date_range = st.sidebar.date_input("Select Date Range", [])
  with col13:
    selected_jobs = st.sidebar.multiselect("Select Job IDs", df_2023['Job'].dropna().unique())
  with col12:
    selected_designation = st.sidebar.multiselect("Select Designations", df_2023['Designation'].dropna().unique())

# Section Header
  st.markdown(
    """
    <h2 style="text-align: center; color: black; margin-bottom: 18px;">
        Project Resource Allocation Dashboard
    </h2>
    """,
    unsafe_allow_html=True,
)
  # Filter Data Based on User Selections
  filtered_df_2023 = df_2023.copy()
  if date_range:
    filtered_df_2023 = filtered_df_2023[(filtered_df_2023['Week End'] >= pd.to_datetime(date_range[0])) &
                                        (filtered_df_2023['Week End'] <= pd.to_datetime(date_range[1]))]
  if selected_jobs:
    filtered_df_2023 = filtered_df_2023[filtered_df_2023['Job'].isin(selected_jobs)]
  if selected_designation:
    filtered_df_2023 = filtered_df_2023[filtered_df_2023['Designation'].isin(selected_designation)]
  #Visuals 
  # Column 1, Row 1: Top Positions by Regular Hours
  top_positions_reg_hours = filtered_df_2023.groupby('Designation')['Reg Hrs'].sum().reset_index().sort_values(by='Reg Hrs', ascending=False).head(5)
  fig = go.Figure()
  fig.add_trace(go.Bar(x=top_positions_reg_hours.sort_values(by='Reg Hrs', ascending=True)['Reg Hrs'],
                     y=top_positions_reg_hours.sort_values(by='Reg Hrs', ascending=True)['Designation'],
                     orientation='h',
                     text=top_positions_reg_hours.sort_values(by='Reg Hrs', ascending=True)['Reg Hrs'],
                     textposition='inside'))

  fig.update_layout(title={'text': f'Top Positions by Regular Hours',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20}
        },
                  paper_bgcolor='#FFFFFF',
                  plot_bgcolor='#FFFFFF',
                  showlegend=False)
  # Column 2, Row 1: Top Positions by Overtime Hours
  top_positions_ot_hours = filtered_df_2023.groupby('Designation')['OT Hrs'].sum().reset_index().sort_values(by='OT Hrs', ascending=False).head(5)
  fig2 = go.Figure()
  fig2.add_trace(go.Bar(x=top_positions_ot_hours.sort_values(by='OT Hrs', ascending=True)['OT Hrs'],
                     y=top_positions_ot_hours.sort_values(by='OT Hrs', ascending=True)['Designation'],
                     orientation='h',
                     text=top_positions_ot_hours.sort_values(by='OT Hrs', ascending=True)['OT Hrs'],
                     textposition='inside',
    insidetextanchor='middle',  # Center the text within the bar
    textangle=0,))


  fig2.update_layout(title={'text': f'Top Positions by Overtime Hours',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20}
        },
                  paper_bgcolor='#FFFFFF',
                  plot_bgcolor='#FFFFFF',
                  showlegend=False)
  
  project_time_by_position  = filtered_df_2023.groupby('Designation').agg(
    total_regular_hours=('Reg Hrs', 'sum'),
    total_ot_hours=('OT Hrs', 'sum'),
    num_projects=('WBS', 'nunique')
).reset_index()
  # Mapping old column names to new names
  column_mapping = {
    "designation": "Designation",
    "total_regular_hours": "Regular Hours",
    "total_ot_hours": "Overtime Hours",
    "num_projects": "Number of Projects"
}

  project_time_by_position.rename(columns=column_mapping, inplace=True)
  project_time_by_position = project_time_by_position.sort_values(by='Regular Hours', ascending=False)
#Calculate the average regular hours per project
  project_time_by_position['Average Per Project'] = round((project_time_by_position['Regular Hours'] / project_time_by_position['Number of Projects']),1)
  fig12 = go.Figure(
    data=[
        go.Table(
             columnwidth=[15, 15, 15, 15, 30],
            header=dict(
                values=list(project_time_by_position.columns),
                fill_color='blue',
                align='center',
                font=dict(size=12, color='white', family='Arial'),
                line_color='darkslategray'
            ),
            cells=dict(
                values=[project_time_by_position[col] for col in project_time_by_position.columns],
                fill_color='white',
                align='center',
                font=dict(size=11, color='black', family='Arial'),
                line_color='darkslategray'
            )
        )
    ]
)

# Update layout for appeal
  fig12.update_layout(title={'text': f'Project Hours Overview by Designation',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20}
        },)
  # Column 1, Row 2: Weekly Trends
  weekly_trends = filtered_df_2023.groupby(['Week End'])[['Reg Hrs', 'OT Hrs']].sum().reset_index()
  fig_weekly_trends = go.Figure()
  fig_weekly_trends.add_trace(go.Scatter(x=weekly_trends['Week End'], y=weekly_trends['Reg Hrs'], mode='lines', name='Regular Hours',  line=dict(color='blue')))
  fig_weekly_trends.add_trace(go.Scatter(x=weekly_trends['Week End'], y=weekly_trends['OT Hrs'], mode='lines', name='Overtime Hours',  line=dict(color='red')))
  # Add Min and Max Annotations for Regular Hours
  min_reg = weekly_trends.loc[weekly_trends['Reg Hrs'].idxmin()]
  max_reg = weekly_trends.loc[weekly_trends['Reg Hrs'].idxmax()]
  fig_weekly_trends.add_annotation(x=min_reg['Week End'], y=min_reg['Reg Hrs'],
                                 text="Min", showarrow=True, arrowhead=2)
  fig_weekly_trends.add_annotation(x=max_reg['Week End'], y=max_reg['Reg Hrs'],
                                 text="Max", showarrow=True, arrowhead=2)
  # Add Min and Max Annotations for Overtime Hours
  min_ot = weekly_trends.loc[weekly_trends['OT Hrs'].idxmin()]
  max_ot = weekly_trends.loc[weekly_trends['OT Hrs'].idxmax()]
  fig_weekly_trends.add_annotation(x=min_ot['Week End'], y=min_ot['OT Hrs'],
                                 text="Min", showarrow=True, arrowhead=2)
  fig_weekly_trends.add_annotation(x=max_ot['Week End'], y=max_ot['OT Hrs'],
                                 text="Max", showarrow=True, arrowhead=2)
  fig_weekly_trends.update_layout(title={'text': f'Weekly Trends',
            'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Center the title horizontally
        'font': {'size': 20}
        },
                  paper_bgcolor='#FFFFFF',
                  plot_bgcolor='#FFFFFF',
                  xaxis_title="Week End",
                  yaxis_title="Hours",
                  showlegend=False)

  filtered_df_2023['Total Hrs'] = filtered_df_2023['Reg Hrs'] + filtered_df_2023['OT Hrs']
  filtered_df_2023['Job'] = filtered_df_2023['Job'].astype(str)
  filtered_df_2023['Total Hrs'] = filtered_df_2023['Total Hrs'].drop_duplicates().dropna().astype('int64')

  # Column 2, Row 2: Monthly Overtime-to-Regular Ratio
  # top_projects = 
  col1, col2 = st.columns([1, 1])
  with col1:
    st.plotly_chart(fig, use_container_width=True)
  with col2:
    st.plotly_chart(fig2, use_container_width=True)
# Row 2: Weekly Trends
  st.plotly_chart(fig_weekly_trends, use_container_width=True)
  st.markdown(
        """
    <div style="padding: 10px; font-size: 14px; line-height: 1.6;">
        <p><strong>Weekly Trends:</strong></p>
        <ul>
        <li>This visual represents the <strong>the regular and overtime hours spent on projects</strong> by the company on a <strong>weekly basis</strong>.</li>
        <li>It includes labels for the <strong>minimum</strong> and <strong>maximum</strong> hours to highlight the weeks with the <strong>least</strong> and <strong>most hours</strong></li>
        <li>Use the filters provided to explore the data in more <strong>detail</strong> and uncover <strong>granular insights</strong>.</li>
        </ul>
    </div>
        """,
        unsafe_allow_html=True,
    )
  st.plotly_chart(fig12, use_container_width=True)


# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
# # Load data
df_jobs = pd.read_csv('job_numbers_23.csv')

# Data preprocessing
df_2023['Week End'] = pd.to_datetime(df_2023['Week End'], errors='coerce')
df_2023['Week'] = df_2023['Week End'].dt.day_of_week
df_2023['Month'] = pd.Categorical(df_2023['Week End'].dt.strftime('%B'), categories=month_order, ordered=True)
df_2023 = df_2023.sort_values(by='Month')

# Merge DataFrames
df_jobs['Job Numbers'] = df_jobs['Job Numbers'].astype(str)
df_projects = df_2023.copy()
df_projects['Job'] = df_projects['Job'].astype(str)
projects_23 = pd.merge(df_jobs, df_projects, left_on='Job Numbers', right_on='Job', how='inner').drop_duplicates()

# Select relevant columns
relevant_columns = [
    'Area (Ha)', 'Number of Services', 'Number of tender Packages', 'Job Numbers',
    'Duration of Work (Weeks)', 'Designation', 'Month', 'Reg Hrs'
]
projects_23 = projects_23[relevant_columns].dropna()
projects_23['Designation'] = projects_23['Designation'].str.lower()

# Cache encoded label encoders and preprocessing
@st.cache_data
def preprocess_data(data):
    label_encoders = {}
    for column in ['Designation', 'Month', 'Job Numbers']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le
    return data, label_encoders

projects_23, label_encoders = preprocess_data(projects_23)

# Cache the model training function
@st.cache_data
def train_model(X, y, n_estimators=200, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# Streamlit Predictive Model Tab
# Streamlit Predictive Model Tab
if tabs == 'Predictive Model':
    st.title("Predict Regular Hours")
    st.sidebar.header("Input Features")

    # Define features and target
    X = projects_23[['Area (Ha)', 'Number of Services', 'Number of tender Packages', 'Job Numbers',
                     'Duration of Work (Weeks)', 'Designation', 'Month']]
    y = projects_23['Reg Hrs']

    # Initialize the session state for the model
    if "rf_model" not in st.session_state:
        st.session_state["rf_model"] = None

    # Train Model Button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            st.session_state["rf_model"] = train_model(X, y, n_estimators=100, test_size=0.2)  # Reduced n_estimators
        st.success("Model trained successfully!")

    # Ensure model is trained before allowing predictions
    if st.session_state["rf_model"] is not None:
        # Sliders for numerical inputs
        area = st.sidebar.slider("Area (Ha)", min_value=50, max_value=500, step=10)
        num_services = st.sidebar.slider("Number of Services", min_value=5, max_value=10, step=1)
        num_tender_packages = st.sidebar.slider("Number of Tender Packages", min_value=1, max_value=12, step=1)
        duration_weeks = st.sidebar.slider("Duration of Work (Weeks)", min_value=10, max_value=200, step=5)

        # Dropdowns for categorical inputs
        designation_options = label_encoders['Designation'].inverse_transform(range(len(label_encoders['Designation'].classes_)))
        job_number_options = label_encoders['Job Numbers'].inverse_transform(range(len(label_encoders['Job Numbers'].classes_)))

        designation = st.sidebar.selectbox("Designation", options=designation_options)
        job_number = st.sidebar.selectbox("Job Numbers", options=job_number_options)
        month = st.sidebar.selectbox("Month", options=month_order)

        # Encode user inputs
        encoded_designation = label_encoders['Designation'].transform([designation])[0]
        encoded_job_number = label_encoders['Job Numbers'].transform([job_number])[0]
        encoded_month = month_order.index(month)  # Use predefined order for months

        # Prepare input data for prediction
        user_input = pd.DataFrame({
            'Area (Ha)': [area],
            'Number of Services': [num_services],
            'Number of tender Packages': [num_tender_packages],
            'Job Numbers': [encoded_job_number],
            'Duration of Work (Weeks)': [duration_weeks],
            'Designation': [encoded_designation],
            'Month': [encoded_month]
        })

        # Predict Button
        if st.button("Predict"):
            prediction = st.session_state["rf_model"].predict(user_input)
            st.write(f"### Predicted Regular Hours: {prediction[0]:.2f}")
    else:
        st.warning("Please train the model before making predictions.")

