import pandas as pd
import numpy as np
import os

# Define Features and Weights (Must match the project logic)
FEATURES = [
    'Fake_Urgency', 'Scarcity', 'Confusing_Text', 'Hidden_Cost',
    'Forced_Action', 'Social_Proof_Fake', 'Misdirection', 'Visual_Trick',
    'Confirmshaming', 'Sneak_Into_Basket', 'Roach_Motel', 'Privacy_Zuckering',
    'Trick_Questions'
]

WEIGHTS = {
    'Fake_Urgency':      0.12,
    'Hidden_Cost':       0.12,
    'Sneak_Into_Basket': 0.12,
    'Roach_Motel':       0.10,
    'Scarcity':          0.08,
    'Privacy_Zuckering': 0.08,
    'Confusing_Text':    0.08,
    'Trick_Questions':   0.08,
    'Forced_Action':     0.06,
    'Misdirection':      0.05,
    'Confirmshaming':    0.05,
    'Social_Proof_Fake': 0.04,
    'Visual_Trick':      0.02,
}

THRESHOLD = 0.50

def generate_synthetic_data(n_samples=2000):
    print(f"Generating {n_samples} synthetic samples...")
    
    # Generate random binary features
    # We want a mix: some dark patterns, some clean
    data = np.random.randint(0, 2, size=(n_samples, len(FEATURES)))
    df = pd.DataFrame(data, columns=FEATURES)
    
    # Add Website Name (for variety)
    websites = ["FashionHub", "TechGadget", "BookWorm", "TravelGo", "HealthCare+", "GameWorld", "HomeDecor", "Foodie", "Marketplace", "QuickShop"]
    df.insert(0, 'Website_Name', [f"{np.random.choice(websites)}_{i}" for i in range(n_samples)])
    
    # Calculate Weighted Score
    weight_array = np.array([WEIGHTS[f] for f in FEATURES])
    df['Weighted_Score'] = df[FEATURES].values @ weight_array
    
    # Assign Labels
    df['Label'] = (df['Weighted_Score'] >= THRESHOLD).astype(int)
    
    # Add some "Ambiguous" samples (close to threshold) to challenge the model
    # (Optional: Not strictly necessary for basic synthetic data, but good for robust ML)
    
    return df

if __name__ == "__main__":
    df_synthetic = generate_synthetic_data(2000)
    
    output_path = "synthetic_dataset.xlsx"
    df_synthetic.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"DONE: Synthetic database created: {output_path}")
    print(f"   Dark Patterns: {sum(df_synthetic['Label'] == 1)}")
    print(f"   Clean Pages:   {sum(df_synthetic['Label'] == 0)}")
