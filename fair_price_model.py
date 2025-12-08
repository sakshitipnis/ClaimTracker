import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_model():
    # Load data
    df = pd.read_csv("historical_claims.csv")
    
    # Features and Target
    X = df[["Disease", "Hospital_Tier"]]
    y = df["Claim_Amount"]
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Disease', 'Hospital_Tier'])
        ])
    
    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Save
    joblib.dump(model, "fair_price_model.pkl")
    print("Model trained and saved as 'fair_price_model.pkl'")
    print(f"Model Score (R2): {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_model()
