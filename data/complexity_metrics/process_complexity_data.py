#!/usr/bin/env python3
"""
Script to process complexity data using the existing process_dataset method from compleximeter.
Converts parquet data into the dataset format expected by compleximeter and adds level information.
"""

import pandas as pd
import numpy as np
from compleximeter import ComplexiMeter
import os
import sys

def load_and_merge_datasets():
    """Load and merge datasets."""
    print("Loading datasets...")
    
    test_df = pd.read_parquet('data/test-00000-of-00001.parquet')
    print(f"Test dataset: {test_df.shape[0]} rows")
    
    validation_df = pd.read_parquet('data/validation-00000-of-00001.parquet')
    print(f"Validation dataset: {validation_df.shape[0]} rows")
    
    merged_df = pd.concat([test_df, validation_df], ignore_index=True)
    print(f"Merged dataset: {merged_df.shape[0]} rows")
    
    return merged_df

def convert_to_compleximeter_dataset(merged_df):
    """Convert dataframe to compleximeter dataset format."""
    print("Converting to compleximeter dataset format...")
    
    dataset = []
    total_rows = len(merged_df)
    
    for idx, row in merged_df.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"Progress: Processed {idx + 1}/{total_rows} rows ({(idx + 1)/total_rows*100:.1f}%)", flush=True)
        
        original_text = row['original']
        simplifications = row['simplifications']
        
        dataset.append({
            'id': f"row_{idx}_level_0",
            'text': original_text,
            'level': 0,
            'row_id': idx
        })
        
        for sim_idx, sim_text in enumerate(simplifications):
            level = sim_idx + 1
            dataset.append({
                'id': f"row_{idx}_level_{level}",
                'text': sim_text,
                'level': level,
                'row_id': idx
            })
    
    print(f"Converted to dataset with {len(dataset)} total texts", flush=True)
    return dataset

def main():
    """Process complexity data using compleximeter."""
    print("Starting complexity data processing...")
    
    try:
        merged_df = load_and_merge_datasets()
        
        dataset = convert_to_compleximeter_dataset(merged_df)
        
        print("Initializing ComplexiMeter...")
        cm = ComplexiMeter()
        
        print("Processing dataset with compleximeter...")
        output_file = 'complexity_results2.csv'
        cm.process_dataset(dataset, output_file)
        
        print("Loading results and adding level information...")
        results_df = pd.read_csv(output_file)
        
        results_df['level'] = results_df['id'].str.extract(r'level_(\d+)').astype(int)
        results_df['row_id'] = results_df['id'].str.extract(r'row_(\d+)_level').astype(int)
        
        results_df = results_df.rename(columns={'input': 'text'})
        
        cols = ['text', 'level', 'row_id'] + [col for col in results_df.columns 
                                              if col not in ['text', 'level', 'row_id', 'id']]
        results_df = results_df[cols]
        
        final_output = 'complexity_results_with_levels.csv'
        results_df.to_csv(final_output, index=False, encoding='utf-8')
        
        print("\nSummary:")
        print(f"Original rows: {len(merged_df)}", flush=True)
        print(f"Total processed texts: {len(results_df)}", flush=True)
        print(f"Expected texts per row: 11 (1 original + 10 simplifications)", flush=True)
        print(f"Levels present: {sorted(results_df['level'].unique())}", flush=True)
        print(f"Results saved to: {final_output}", flush=True)
        
        print("\nSample results:")
        print(results_df[['text', 'level', 'row_id']].head(20))
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 