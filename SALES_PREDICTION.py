# Import necessary libraries
import os  # Operating system dependent functionality
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Linear regression models
from sklearn.tree import DecisionTreeRegressor  # Decision tree regressor
from sklearn.ensemble import RandomForestRegressor  # Random forest regressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Model selection and evaluation tools
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance

# Load the advertising data from the CSV file
file_path = r'C:\Users\Admin\Downloads\ENCRYPTIX\SALES PREDICTION USING PYTHON\advertising.csv'  # Path to the CSV file
data = pd.read_csv(file_path)  # Read the CSV file into a pandas DataFrame

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(data.isnull().sum())  # Print the sum of missing values in each column

# Visualize the relationship between sales and advertising mediums using Plotly Express
import plotly.express as px  # Interactive visualization library

# Scatter plot with trendline for TV advertising
figure = px.scatter(data_frame=data, x="TV", y="Sales", size="TV", trendline="ols",  color="Sales",
                    title="TV Advertising vs Sales")  # Create scatter plot using Plotly Express
figure.show()  # Display the plot

# Scatter plot with trendline for Newspaper advertising
figure = px.scatter(data_frame=data, x="Newspaper", y="Sales", size="Newspaper", trendline="ols",  color="Sales",
                    title="Newspaper Advertising vs Sales")  # Create scatter plot using Plotly Express
figure.show()  # Display the plot

# Scatter plot with trendline for Radio advertising
figure = px.scatter(data_frame=data, x="Radio", y="Sales", size="Radio", trendline="ols",  color="Sales", 
                    title="Radio Advertising vs Sales")  # Create scatter plot using Plotly Express
figure.show()  # Display the plot

# Calculate correlation between features and target variable (Sales)
correlation = data.corr()  # Compute pairwise correlation of columns
print("Correlation with Sales:")
print(correlation["Sales"].sort_values(ascending=False))  # Print correlation with Sales variable

# Feature Engineering: Create interaction features
data['TV_Radio'] = data['TV'] * data['Radio']  # Create interaction feature between TV and Radio
data['TV_Newspaper'] = data['TV'] * data['Newspaper']  # Create interaction feature between TV and Newspaper
data['Radio_Newspaper'] = data['Radio'] * data['Newspaper']  # Create interaction feature between Radio and Newspaper

# Prepare the features (advertising expenditures) and target variable (Sales) for modeling
X = data[["TV", "Newspaper", "Radio", "TV_Radio", "TV_Newspaper", "Radio_Newspaper"]].values  # Features
y = data["Sales"].values  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into train and test sets

# Feature Scaling
# Initialize a StandardScaler object to scale features
scaler = StandardScaler()

# Scale the training features (X_train)
X_train = scaler.fit_transform(X_train)

# Scale the testing features (X_test)
X_test = scaler.transform(X_test)

# Train and evaluate multiple models
# Define a dictionary containing different regression models to be evaluated
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# Loop through each model in the dictionary
for name, model in models.items():
    # Perform cross-validation
    # Calculate R-squared scores using cross-validation on the training data
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # Print the mean R-squared score across all cross-validation folds
    print(f"{name} Cross-Validation R-squared scores: {scores.mean()}")

    # Train the model
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the testing set
    # Calculate R-squared score on the testing data
    score = model.score(X_test, y_test)
    # Print the R-squared score on the testing set
    print(f"{name} R-squared score on the testing set: {score}")

# Hyperparameter Tuning for Random Forest
# Define a grid of hyperparameters to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search using Random Forest regressor
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2', n_jobs=-1)
# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by grid search
print("Best parameters for Random Forest:", grid_search.best_params_)
# Get the best Random Forest model
best_rf_model = grid_search.best_estimator_

# Evaluate the best model on the testing set
# Calculate R-squared score on the testing set using the best model
best_rf_score = best_rf_model.score(X_test, y_test)
# Print the R-squared score on the testing set after tuning
print("Random Forest R-squared score on the testing set after tuning:", best_rf_score)

# Predict sales for new advertising expenditures
# Define new advertising features
new_features = np.array([[230.1, 37.8, 69.2, 230.1*69.2, 230.1*37.8, 69.2*37.8]])
# Scale the new features
new_features_scaled = scaler.transform(new_features)
# Predict sales using the best Random Forest model
predicted_sales = best_rf_model.predict(new_features_scaled)
# Print the predicted sales for new advertising expenditures
print("Predicted sales for new advertising expenditures:", predicted_sales)

# Print details about the importance of sales prediction and the role of a Data Scientist
print("\nSales prediction guides business decisions by forecasting customer purchases, driven by data analysis and machine learning techniques. Data Scientists play a pivotal role in optimizing advertising strategies, maximizing sales potential.")