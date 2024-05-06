
import numpy as np
import pandas as pd

def generate_data(num_samples=1000):
    # Seed for reproducibility
    np.random.seed(42)

    # Features
    x1 = np.linspace(0, 24, num_samples)  # Time of day from 0 to 24 hours
    x2 = np.abs(np.sin(np.linspace(0, 10, num_samples)) + np.random.normal(0, 0.1, num_samples)) * 100  # Fluctuations with noise
    x3 = np.random.poisson(lam=20, size=num_samples)  # Number of transactions/connections

    # Target variable: Modify this formula to fit the predictive model needs
    y = 3*x1 + 2*x2 + 5*x3 + np.random.normal(0, 10, num_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })

    # Output CSV file
    data.to_csv('synthetic_data.csv', index=False)
    print("Data generated and saved to synthetic_data.csv.")

if __name__ == "__main__":
    generate_data()  # Call the function with default or modified parameters
