import pandas as pd
import numpy as np

# Load the CSV file; adjust the filename as necessary.
filename = "generated_dataset (8).csv"
df = pd.read_csv(filename)

# Test 1: Check the shape of the DataFrame
expected_shape = (400, 15)
if df.shape == expected_shape:
    print("Test 1 Passed: DataFrame shape is correct.")
else:
    print(f"Test 1 Failed: Expected shape {expected_shape}, but got {df.shape}.")

# Test 2: Check column labels (should be A through O)
expected_columns = list("ABCDEFGHIJKLMNO")
if list(df.columns) == expected_columns:
    print("Test 2 Passed: Column labels are correct.")
else:
    print(f"Test 2 Failed: Expected columns {expected_columns}, but got {list(df.columns)}.")

# Test 3: Check that each variable's values range between 1 and 5
all_in_range = True
for col in df.columns:
    col_min, col_max = df[col].min(), df[col].max()
    if col_min < 1 or col_max > 5:
        print(f"Test 3 Failed: Column {col} has values outside the range 1-5 (min: {col_min}, max: {col_max}).")
        all_in_range = False
if all_in_range:
    print("Test 3 Passed: All column values are within the range 1 to 5.")

# Define groups based on the specification
group_ABCD = df[['A', 'B', 'C', 'D']]
group_EFGH = df[['E', 'F', 'G', 'H']]
group_IJKL = df[['I', 'J', 'K', 'L']]
group_MNO  = df[['M', 'N', 'O']]

# Test 4: Check high internal correlation for each group (r >= 0.8)
def check_internal_correlation(group, group_name, threshold=0.8):
    corr_matrix = group.corr()
    passed = True
    # Check each unique pair
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val < threshold:
                print(f"Test 4 Failed: In group {group_name}, correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]} is {corr_val:.2f} (< {threshold}).")
                passed = False
    if passed:
        print(f"Test 4 Passed: All pairwise correlations in group {group_name} are >= {threshold}.")

check_internal_correlation(group_ABCD, "A, B, C, D")
check_internal_correlation(group_EFGH, "E, F, G, H")
check_internal_correlation(group_IJKL, "I, J, K, L")
check_internal_correlation(group_MNO,  "M, N, O")

# Test 5: Check that between-group correlations are comparatively low (absolute correlation < 0.3)
def check_between_groups(group1, group2, group1_name, group2_name, max_threshold=0.3):
    # Create a correlation matrix between the two groups.
    # We compute the correlation between each column in group1 and each column in group2.
    group1_values = group1.to_numpy().T  # shape: (num_vars1, samples)
    group2_values = group2.to_numpy().T  # shape: (num_vars2, samples)
    # Calculate the correlation coefficient matrix manually
    corr_matrix = np.corrcoef(group1_values, group2_values)[0:group1.shape[1], group1.shape[1]:]
    corr_df = pd.DataFrame(corr_matrix, index=group1.columns, columns=group2.columns)
    max_corr = np.abs(corr_df).max().max()
    if max_corr < max_threshold:
        print(f"Test 5 Passed: Maximum absolute correlation between groups {group1_name} and {group2_name} is {max_corr:.2f} (< {max_threshold}).")
    else:
        print(f"Test 5 Failed: Maximum absolute correlation between groups {group1_name} and {group2_name} is {max_corr:.2f} (>= {max_threshold}).")

check_between_groups(group_ABCD, group_EFGH, "A, B, C, D", "E, F, G, H")
check_between_groups(group_IJKL, group_MNO, "I, J, K, L", "M, N, O")

# Optionally, you could add tests for normality (for example, using a Shapiro-Wilk test)
# However, note that with 400 samples, even slight deviations from normality may be flagged.
# For brevity, normality testing is omitted here.
