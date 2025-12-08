import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_fraud_model():
    # Load data
    df = pd.read_csv("historical_claims.csv")
    
    # Features and Target
    X = df[["Disease", "Hospital_Tier", "Claim_Amount"]]
    y = df["Fraud"]
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Disease', 'Hospital_Tier'])
        ],
        remainder='passthrough' # Keep Claim_Amount as is
    )
    
    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Save
    joblib.dump(model, "fraud_model.pkl")
    print("Fraud Model trained and saved as 'fraud_model.pkl'")
    print(f"Model Accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_fraud_model()
