# Score Sorcery: Math Predictions & EDA Marvels

### Setup
a) Run `pip install -r requirements.txt`

b) Run `python app.py`

c) Outputs the R2 score amongst the models deployed.

d) Visit the `notebook` folder to directly access `EDA` and `Model Training` files. 

---

### Deployed App

You can access the deployed app here: [Math Score Predictor](https://math-score-predictor.streamlit.app/)

### YouTube Presentation

[Student Performance Indicator](https://bit.ly/Student_Performance_Indicator)

---

| **Dataset**                 |                        |
|-----------------------------|------------------------|
| **Categorical Variables**   | **Numerical Variables**|
| - Gender                    | - Math Score           |
| - Race/Ethnicity            | - Reading Score        |
| - Parental Level of Education | - Writing Score      |
| - Lunch                     |                        |
| - Test Preparation Course   |                        |

### Objective
To predict the Math Score by implementing a Modular Structure where data has been ingested using an SQL database. We performed Data Transformation and Model Training. Additionally, modules for Exception Handling and Logging were implemented to track the work progress.

### Method
Performed One-Hot Encoding on categorical variables and Standard Scaling on numerical variables.

### Models
- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression
- XGBRegressor
- CatBoosting Regressor
- AdaBoost Regressor

---
