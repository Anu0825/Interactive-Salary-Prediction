import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a sample dataset with 50 rows
def generate_sample_data():
    np.random.seed(42)
    years_experience = np.random.uniform(0, 10, 50)
    salary = 30000 + (years_experience * 8000) + np.random.normal(0, 5000, 50)
    data = pd.DataFrame({"YearsExperience": years_experience, "Salary": salary})
    return data

# Main function
def main():
    st.set_page_config(page_title="Interactive Salary Prediction", layout="wide")

    st.title("Salary Prediction App")
    st.markdown(
        "This app predicts salaries based on years of experience using a linear regression model."
    )

    # Load data
    st.sidebar.header("Upload or Use Sample Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = generate_sample_data()

    st.write("### Dataset", data.head())

    # Visualize data
    st.write("### Data Visualization")
    fig, ax = plt.subplots(figsize=(6, 4)) 
    sns.scatterplot(x="YearsExperience", y="Salary", data=data, ax=ax, color="blue", s=50)
    ax.set_title("Salary vs. Years of Experience", fontsize=14)
    ax.set_xlabel("Years of Experience", fontsize=12)
    ax.set_ylabel("Salary", fontsize=12)
    sns.despine()
    st.pyplot(fig)

    X = data[["YearsExperience"]]
    y = data["Salary"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"### Model Evaluation\nMean Squared Error: {mse:.2f}")

    # Sidebar: User input for prediction
    st.sidebar.header("Make a Prediction")
    years_experience = st.sidebar.slider("Years of Experience", min_value=0.0, max_value=10.0, step=0.1)

    if st.sidebar.button("Predict Salary"):
        salary_pred = model.predict([[years_experience]])[0]
        st.sidebar.write(f"### Predicted Salary: ${salary_pred:.2f}")

    # Display regression line
    st.write("### Regression Line")
    data["PredictedSalary"] = model.predict(data[["YearsExperience"]])
    fig, ax = plt.subplots(figsize=(6, 4)) 
    sns.scatterplot(x="YearsExperience", y="Salary", data=data, ax=ax, color="blue", s=50, label="Actual")
    sns.lineplot(x="YearsExperience", y="PredictedSalary", data=data, ax=ax, color="red", label="Prediction")
    ax.set_title("Regression Line", fontsize=14)
    ax.set_xlabel("Years of Experience", fontsize=12)
    ax.set_ylabel("Salary", fontsize=12)
    sns.despine()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
