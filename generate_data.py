import pandas as pd
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=1000):
    diseases = [
        "Flu", "Heart Attack", "Broken Leg", "Diabetes", "Hypertension", 
        "Appendicitis", "Cataract", "Pneumonia", "Covid-19", "Migraine"
    ]
    
    hospital_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    
    # Base costs for diseases (in INR)
    base_costs = {
        "Flu": 2000, "Heart Attack": 300000, "Broken Leg": 150000, 
        "Diabetes": 5000, "Hypertension": 3000, "Appendicitis": 200000, 
        "Cataract": 40000, "Pneumonia": 50000, "Covid-19": 100000, "Migraine": 2000
    }
    
    data = []
    
    start_date = datetime(2023, 1, 1)
    
    for _ in range(num_records):
        disease = random.choice(diseases)
        tier = random.choice(hospital_tiers)
        
        # Calculate cost with variance
        base = base_costs[disease]
        tier_multiplier = {"Tier 1": 1.5, "Tier 2": 1.2, "Tier 3": 1.0}[tier]
        
        # Introduce Fraud Patterns
        is_fraud = 0
        variance = random.uniform(0.8, 1.2) # Normal variance
        
        # 1. Inflated Cost Fraud (randomly spike the cost)
        if random.random() < 0.05: # 5% chance of fraud
            variance = random.uniform(1.5, 3.0)
            is_fraud = 1
            
        # 2. Mismatched Tier Fraud (Tier 3 charging like Tier 1)
        if tier == "Tier 3" and random.random() < 0.05:
             tier_multiplier = 1.5 # Charging Tier 1 price
             is_fraud = 1
        
        claim_amount = round(base * tier_multiplier * variance, 2)
        
        # Determine status
        if is_fraud:
            status = "Rejected"
        elif variance > 1.15:
             status = "Rejected" # Rejected for high cost but maybe not fraud
        else:
            status = "Approved"
            
        date = start_date + timedelta(days=random.randint(0, 365))
        
        data.append({
            "Claim_ID": f"CLM{random.randint(10000, 99999)}",
            "Disease": disease,
            "Hospital_Tier": tier,
            "Claim_Amount": claim_amount,
            "Status": status,
            "Fraud": is_fraud,
            "Date": date.strftime("%Y-%m-%d")
        })
        
    df = pd.DataFrame(data)
    df.to_csv("historical_claims.csv", index=False)
    print("Successfully generated 'historical_claims.csv' with 1000 records.")

if __name__ == "__main__":
    generate_synthetic_data()
