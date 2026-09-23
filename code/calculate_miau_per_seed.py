"""
Calculate MIAU (Membership Inference Attack Unlearning) scores per seed.

FIXED:
- compute_fi now returns 1.0 when B==R (ideal case) instead of 0
- This correctly scores an ideal unlearning method at ~99.9 instead of ~66.6
"""

import pandas as pd
import numpy as np
import argparse
import os


def compute_fi(B, R, M):
    """
    Compute f_i score for MIA metric.
    
    FIXED: When B == R (baseline equals retrain), the method achieves ideal
    unlearning behavior. In this case, we should return 1.0 (perfect score)
    not 0 (which would incorrectly penalize perfect unlearning).
    
    Args:
        B: Baseline MIA score
        R: Retrain MIA score  
        M: Method MIA score
    
    Returns:
        f_i score in range [-inf, 1], where 1 is ideal
    """
    denom = abs(B - R)
    if denom < 1e-9:  # B approximately equals R
        # If M is also close to R, this is ideal unlearning
        if abs(M - R) < 1e-9:
            return 1.0  # FIXED: Return 1.0 for ideal case
        else:
            # M differs from R even though B==R, which is unusual
            # Return negative to indicate method is worse than ideal
            return -abs(M - R)
    return (abs(B - R) - abs(M - R)) / denom


def compute_musi(f_i, alpha=13.8):
    """
    Compute MUS_i (Membership Unlearning Score) from f_i.
    
    Uses sigmoid function to map f_i to [0, 100] range.
    """
    return 100 * (1 / (1 + np.exp(-alpha * (f_i - 0.5))))


def calculate_miau_for_csv(input_csv_path, output_csv_path=None):
    """
    Calculate MIAU scores for a compiled results CSV.
    
    Args:
        input_csv_path: Path to compiled_results.csv
        output_csv_path: Path for output (default: adds _MIAU suffix)
    
    Returns:
        DataFrame with MIAU scores added
    """
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_MIAU{ext}"
    
    df = pd.read_csv(input_csv_path)
    
    mia_columns = [
        "Forget vs Retain Membership Inference Attack (MIA)",
        "Forget vs Test Membership Inference Attack (MIA)",
        "Test vs Retain Membership Inference Attack (MIA)"
    ]
    
    # Filter to get baseline and retrain rows
    baseline_df = df[df['unlearning'] == 'baseline']
    retrain_df = df[df['unlearning'] == 'retrain']
    
    final_df = df.copy()
    
    for col in mia_columns:
        f_col = f"f_i ({col})"
        mus_col = f"MUS_i ({col})"
        
        def calculate(row):
            # Find matching baseline
            cond = (baseline_df['seed'] == row['seed']) & \
                   (baseline_df['dataset'] == row['dataset']) & \
                   (baseline_df['model'] == row['model'])
            B_row = baseline_df[cond]
            B = B_row[col].values[0] if not B_row.empty else np.nan
            
            # Find matching retrain
            cond = (retrain_df['seed'] == row['seed']) & \
                   (retrain_df['dataset'] == row['dataset']) & \
                   (retrain_df['model'] == row['model'])
            R_row = retrain_df[cond]
            R = R_row[col].values[0] if not R_row.empty else np.nan
            
            M = row[col]
            if pd.isna(B) or pd.isna(R) or pd.isna(M):
                return pd.Series([np.nan, np.nan])
            
            f_i = compute_fi(B, R, M)
            mus_i = compute_musi(f_i)
            return pd.Series([f_i, mus_i])
        
        final_df[[f_col, mus_col]] = final_df.apply(calculate, axis=1)
    
    # Calculate overall MIAU as mean of MUS_i scores
    mus_cols = [f"MUS_i ({col})" for col in mia_columns]
    final_df["MIAU"] = final_df[mus_cols].mean(axis=1)
    
    # Save results
    final_df.to_csv(output_csv_path, index=False)
    print(f"Saved MIAU results to: {output_csv_path}")
    
    # Print summary
    print("\nMIAU Summary:")
    print(final_df[["Filename", "unlearning", "seed", "MIAU"] + mus_cols].head(20))
    
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MIAU scores from compiled results")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to compiled_results.csv")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output path (default: input_MIAU.csv)")
    args = parser.parse_args()
    
    calculate_miau_for_csv(args.input, args.output)
