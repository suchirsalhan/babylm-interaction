#!/usr/bin/env python3
"""
Chunked version of compleximeter.py for parallel processing on Euler cluster.
Supports processing data in chunks to enable job arrays.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from compleximeter import ComplexiMeter

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

def convert_to_compleximeter_dataset(merged_df, chunk_id, total_chunks):
    """Convert dataframe to compleximeter dataset format."""
    print(f"Converting to compleximeter dataset format for chunk {chunk_id}/{total_chunks}...")
    
    dataset = []
    total_rows = len(merged_df)
    
    chunk_size = total_rows // total_chunks
    start_idx = (chunk_id - 1) * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < total_chunks else total_rows
    
    print(f"Processing rows {start_idx} to {end_idx} (chunk {chunk_id}/{total_chunks})")
    
    for idx in range(start_idx, end_idx):
        row = merged_df.iloc[idx]
        
        if (idx - start_idx + 1) % 100 == 0:
            print(f"Progress: Processed {idx - start_idx + 1}/{end_idx - start_idx} rows in chunk", flush=True)
        
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
    
    print(f"Converted to dataset with {len(dataset)} total texts for chunk {chunk_id}", flush=True)
    return dataset

def main():
    """Process complexity data using compleximeter."""
    parser = argparse.ArgumentParser(description='Process complexity data in chunks')
    parser.add_argument('--chunk-id', type=int, required=True, help='Chunk ID (1-based)')
    parser.add_argument('--total-chunks', type=int, required=True, help='Total number of chunks')
    
    args = parser.parse_args()
    
    print(f"Starting complexity data processing for chunk {args.chunk_id}/{args.total_chunks}...")
    
    try:
        merged_df = load_and_merge_datasets()
        
        dataset = convert_to_compleximeter_dataset(merged_df, args.chunk_id, args.total_chunks)
        
        if not dataset:
            print("No data to process for this chunk")
            return
        
        print("Initializing ComplexiMeter...")
        cm = ComplexiMeter()
        
        print("Processing dataset with compleximeter...")
        output_file = f'complexity_results_without_levels/complexity_results_chunk_{args.chunk_id}.csv'
        cm.process_dataset(dataset, output_file)
        
        print("Loading results and adding level information...")
        results_df = pd.read_csv(output_file)
        
        results_df['level'] = results_df['id'].str.extract(r'level_(\d+)').astype(int)
        results_df['row_id'] = results_df['id'].str.extract(r'row_(\d+)_level').astype(int)
        
        results_df = results_df.rename(columns={'input': 'text'})
        
        cols = ['text', 'level', 'row_id'] + [col for col in results_df.columns 
                                              if col not in ['text', 'level', 'row_id', 'id']]
        results_df = results_df[cols]
        
        final_output = f'complexity_results_with_levels/complexity_results_chunk_{args.chunk_id}_with_levels.csv'
        results_df.to_csv(final_output, index=False, encoding='utf-8')
        
        print(f"\nSummary for chunk {args.chunk_id}:")
        print(f"Processed texts: {len(results_df)}", flush=True)
        print(f"Levels present: {sorted(results_df['level'].unique())}", flush=True)
        print(f"Results saved to: {final_output}", flush=True)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 