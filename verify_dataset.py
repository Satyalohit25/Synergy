import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def verify_dataset(file_path):
    """
    Verifies if the dataset meets the specified requirements:
    1. Sample size of 400
    2. Variables A-D highly correlated (r > 0.8)
    3. Variables E-H highly correlated (r > 0.8)
    4. Variables I-L highly correlated (r > 0.8)
    5. Variables M-O highly correlated (r > 0.8)
    6. Low correlation between groups
    7. Values between 1-5
    8. Normal distribution for all variables
    
    Parameters:
    file_path (str): Path to CSV file containing the dataset
    
    Returns:
    dict: Results of verification tests
    """
    # Load the data
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)  # Changed from read_excel to read_csv
    except Exception as e:
        return {"success": False, "error": f"Failed to load file: {str(e)}"}
    
    # Check if data has 15 columns labeled A to O
    expected_columns = list("ABCDEFGHIJKLMNO")
    if not all(col in df.columns for col in expected_columns):
        return {"success": False, "error": f"Data must contain columns labeled A through O. Found columns: {list(df.columns)}"}
    
    # Use only the specified columns in the expected order
    df = df[expected_columns]
    
    # Check sample size
    sample_size = len(df)
    sample_size_check = sample_size == 400
    print(f"Sample size: {sample_size} (Expected: 400) - {'PASSED' if sample_size_check else 'FAILED'}")
    
    # Check value range (1-5)
    min_val = df[expected_columns].min().min()
    max_val = df[expected_columns].max().max()
    value_range_check = (min_val >= 1) and (max_val <= 5)
    print(f"Value range: {min_val:.2f} to {max_val:.2f} (Expected: 1 to 5) - {'PASSED' if value_range_check else 'FAILED'}")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Define groups
    groups = {
        "Group 1": ["A", "B", "C", "D"],
        "Group 2": ["E", "F", "G", "H"],
        "Group 3": ["I", "J", "K", "L"],
        "Group 4": ["M", "N", "O"]
    }
    
    # Check correlations within and between groups
    within_group_corrs = {}
    between_group_corrs = {}
    
    for group_name, group_vars in groups.items():
        # Within group correlations
        group_corr = corr_matrix.loc[group_vars, group_vars]
        # Remove diagonal (self-correlations)
        mask = ~np.eye(group_corr.shape[0], dtype=bool)
        within_group_corrs[group_name] = group_corr.values[mask]
        
        # Between group correlations
        other_groups_vars = [var for var_list in [groups[g] for g in groups if g != group_name] for var in var_list]
        between_group_corrs[group_name] = corr_matrix.loc[group_vars, other_groups_vars].values.flatten()
    
    # Prepare results
    correlation_results = {}
    for group_name in groups:
        within_min = np.min(within_group_corrs[group_name])
        within_mean = np.mean(within_group_corrs[group_name])
        within_high_check = within_min > 0.8
        
        between_max = np.max(np.abs(between_group_corrs[group_name]))
        between_mean = np.mean(np.abs(between_group_corrs[group_name]))
        between_low_check = between_max < 0.8
        
        correlation_results[group_name] = {
            "within_min": within_min,
            "within_mean": within_mean,
            "within_high_check": within_high_check,
            "between_max": between_max,
            "between_mean": between_mean,
            "between_low_check": between_low_check
        }
        
        print(f"{group_name} internal correlation: min={within_min:.3f}, mean={within_mean:.3f} (Expected: >0.8) - {'PASSED' if within_high_check else 'FAILED'}")
        print(f"{group_name} external correlation: max={between_max:.3f}, mean={between_mean:.3f} (Expected: <0.8) - {'PASSED' if between_low_check else 'FAILED'}")
    
    # Check for normality (Shapiro-Wilk test)
    normality_results = {}
    for col in expected_columns:
        stat, p = stats.shapiro(df[col])
        is_normal = p > 0.05
        normality_results[col] = {
            "p_value": p,
            "is_normal": is_normal
        }
        print(f"Normality test for {col}: p={p:.4f} - {'PASSED' if is_normal else 'FAILED'}")
    
    # Prepare results dictionary
    results = {
        "success": True,
        "sample_size": {
            "value": sample_size,
            "passed": sample_size_check
        },
        "value_range": {
            "min": min_val,
            "max": max_val,
            "passed": value_range_check
        },
        "correlation": correlation_results,
        "normality": normality_results
    }
    
    # Create visualization directory
    viz_dir = "correlation_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate heatmap visualization
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/correlation_heatmap.png")
    
    # Generate normality plots for each variable
    for col in expected_columns:
        plt.figure(figsize=(10, 4))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(df[col], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {col}")
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/normality_{col}.png")
        plt.close()
    
    # Generate group correlation visualizations
    for group_name, group_vars in groups.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[group_vars].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
        plt.title(f"Correlation within {group_name}")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/group_correlation_{group_name}.png")
        plt.close()
    
    print(f"\nResults summary:")
    print(f"- Sample size check: {'PASSED' if sample_size_check else 'FAILED'}")
    print(f"- Value range check: {'PASSED' if value_range_check else 'FAILED'}")
    
    correlation_check_passed = all(result["within_high_check"] and result["between_low_check"] for result in correlation_results.values())
    print(f"- Correlation structure check: {'PASSED' if correlation_check_passed else 'FAILED'}")
    
    normality_check_passed = all(result["is_normal"] for result in normality_results.values())
    print(f"- Normality check: {'PASSED' if normality_check_passed else 'FAILED'}")
    
    overall_passed = sample_size_check and value_range_check and correlation_check_passed and normality_check_passed
    print(f"\nOverall assessment: {'PASSED' if overall_passed else 'FAILED'}")
    print(f"\nCheck the '{viz_dir}' folder for detailed visualizations.")
    
    return results

# Example usage
if __name__ == "__main__":
    file_path = "generated_dataset (6).csv"
    verify_dataset(file_path)