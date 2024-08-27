import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Set up the page configuration
st.set_page_config(
    page_title="Math Score Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load the dataset
df = pd.read_csv('data/raw.csv')

# Prepare the data
X = df.drop(columns=['math_score'], axis=1)
y = df['math_score']

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),        
    ]
)

X = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# App Title and Description
st.title("📊 Math Score Prediction")
st.markdown("""
    Welcome to the **Math Score Predictor**! This app uses a machine learning model to predict a student's math score based on various features. 
    Simply enter the required information below, and the app will predict the likely math score.
""")

st.markdown("---")

# Sidebar for model performance metrics
st.sidebar.header("Model Performance")
st.sidebar.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
st.sidebar.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
st.sidebar.write(f"**Root Mean Squared Error:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

st.sidebar.markdown("---")
st.sidebar.write("**Model Details:**")
st.sidebar.markdown("""
**Linear Regression** came out to be the best model among:
- Linear Regression
- Lasso
- Ridge
- K-Neighbors Regressor
- Decision Tree
- Random Forest Regressor
- XGBRegressor
- CatBoost Regressor
- AdaBoost Regressor
""")

st.subheader("Predict New Math Score")
st.markdown("Please fill in the information below:")

with st.form(key="prediction_form"):
    gender = st.selectbox("Gender", ["Select", "female", "male"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["Select", "group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", [
        "Select", "bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"
    ])
    lunch = st.selectbox("Lunch", ["Select", "standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["Select", "none", "completed"])
    reading_score = st.slider("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.slider("Writing Score", min_value=0, max_value=100, value=50)

    submit_button = st.form_submit_button(label="Predict Math Score")

# Process input and display prediction
if submit_button:
    if (gender != 'Select' and race_ethnicity != 'Select' and 
        parental_level_of_education != 'Select' and lunch != 'Select' and 
        test_preparation_course != 'Select'):
        
        # Organize inputs into a dataframe
        input_data = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [parental_level_of_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation_course],
            "reading_score": [reading_score],
            "writing_score": [writing_score]
        })
        
        # Preprocess input data
        input_transformed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_transformed)
        st.success(f"🎯 **Predicted Math Score: {prediction[0]:.2f}**")
    else:
        st.error("Please select a valid option for all fields.")

# Button to show/hide graph
show_graph = st.checkbox('Show Actual vs Predicted Values', value=False)

if show_graph:
    st.subheader("Actual vs Predicted Values")
    st.markdown("The plot below shows the comparison between actual and predicted math scores:")

    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size to make it smaller
    sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={"color": "blue"}, line_kws={"color": "green"}, ax=ax)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Actual vs Predicted Math Scores")
    st.pyplot(fig)

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("### About this App")
st.sidebar.markdown("""
    This application was built to demonstrate how machine learning models can be used for predictive analysis.
""")
