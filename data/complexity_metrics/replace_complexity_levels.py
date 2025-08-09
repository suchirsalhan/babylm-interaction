#!/usr/bin/env python3
import pandas as pd
import sys

def replace_complexity_levels(input_file, output_file):
    """Replace complexity level text labels with numeric values."""
    complexity_mapping = {
        'child': 1,
        'teen': 2, 
        'college student': 3,
        'graduate student': 4,
        'expert': 5
    }
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Replacing complexity levels...")
    df['complexity'] = df['complexity'].map(complexity_mapping)
    

    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Successfully processed {input_file} -> {output_file}")

def main():
    print("Processing individual utterances file...")
    replace_complexity_levels(
        'individual_utterances_complexity_results.csv',
        'individual_utterances_complexity_results_numeric.csv'
    )
    
    print("\nProcessing dialog aggregations file...")
    replace_complexity_levels(
        'dialog_aggregations_complexity_results.csv',
        'dialog_aggregations_complexity_results_numeric.csv'
    )
    
    print("\nDone! Files have been created with numeric complexity levels:")
    print("- individual_utterances_complexity_results_numeric.csv")
    print("- dialog_aggregations_complexity_results_numeric.csv")
    print("\nMapping used:")
    print("1 = child")
    print("2 = teen") 
    print("3 = college student")
    print("4 = graduate student")
    print("5 = expert")

if __name__ == "__main__":
    main() 