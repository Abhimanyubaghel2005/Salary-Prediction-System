import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("data/Employees.csv", sep=None, engine="python")
data = data.drop_duplicates()

# Select features
X = data[["Department", "Years", "Job Rate"]]
y = data["Monthly Salary"]

# Convert text to numbers
le_department = LabelEncoder()
le_job = LabelEncoder()

X["Department"] = le_department.fit_transform(X["Department"])
X["Job Rate"] = le_job.fit_transform(X["Job Rate"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model & encoders
pickle.dump(model, open("salary_model.pkl", "wb"))
pickle.dump(le_department, open("dept_encoder.pkl", "wb"))
pickle.dump(le_job, open("job_encoder.pkl", "wb"))

print("Model trained and saved successfully!")