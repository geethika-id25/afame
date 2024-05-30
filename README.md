# Import libraries (replace with your preferred choices)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Replace with your regression model
# Load data
data = pd.read_csv("movie_data.csv")  # Replace with your data path
# Handle missing values (choose your approach)
# Option 1: Remove rows with missing values for crucial features
data.dropna(subset=["rating", "genre", "director"], inplace=True)
# Option 2: Impute missing values (example: using average rating by genre)
# genre_avg_rating = data.groupby("genre")["rating"].mean()
# data["rating"].fillna(genre_avg_rating, inplace=True)
# Feature engineering
# One-hot encode genres
genre_encoder = OneHotEncoder(sparse=False)
genres = pd.get_dummies(data["genre"])
data = pd.concat([data, genres], axis=1)
data.drop("genre", axis=1, inplace=True)
# Encode director and actors (replace with your chosen method)
# ... (e.g., one-hot encoding or embedding)
# Separate features and target variable
X = data.drop("rating", axis=1)  # Features
y = data["rating"]  # Target variable
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Build and train the model (replace with your chosen model)
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on test data
y_pred = model.predict(X_test)
# Evaluate model performance (replace with your chosen metrics)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")


