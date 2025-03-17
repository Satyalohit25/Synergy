import numpy as np
import pandas as pd

def create_dataset(num_rows=500):
    # Generate Column A as integers in [1, 100]
    col_A = np.random.randint(1, 101, size=num_rows)
    mean_A = np.mean(col_A)
    std_A = np.std(col_A)
    
    # Generate Column B base values as integers in [1, 10]
    col_B_base = np.random.randint(1, 11, size=num_rows)
    mean_A = np.mean(col_A)
    std_A = np.std(col_A)
    # Impact: Final Column B = Base_B * (Column A / mean(Column A))
    # (Impact strength of 2.0 cancels with the normalization step)
    impact_multiplier = col_A / mean_A
    col_B = col_B_base * impact_multiplier
    
    # Generate Column C as integers in [0, 50]
    col_C = np.random.randint(0, 51, size=num_rows)
    
    # Create the DataFrame
    df = pd.DataFrame({
        "Column A": col_A,
        "Column B": col_B,
        "Column C": col_C
    })
    
    return df, mean_A, std_A

# Create dataset
df, mean_A, std_A = create_dataset(500)

# Display the results
print("Dataset preview:")
print(df.head(30))
print("\nSummary for Column A:")
print(f"Mean: {mean_A:.2f}, Std Dev: {std_A:.2f}")
print("\nDataset Description:")
print(df.describe())
