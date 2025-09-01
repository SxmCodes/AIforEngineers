import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('data/telco_churn.csv')

# Pre-processing (same as before)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
for col in categorical_cols:
    if col != 'Churn':
        df[col] = le.fit_transform(df[col])
df['Churn'] = le.fit_transform(df['Churn'])
df.drop('customerID', axis=1, inplace=True)

# Split and train
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional print)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model
joblib.dump(model, 'models/churn_model.pkl')