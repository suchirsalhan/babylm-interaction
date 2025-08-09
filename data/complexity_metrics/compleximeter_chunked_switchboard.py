#!/usr/bin/env python3
"""
Chunked version of compleximeter.py for processing switchboard speaker level data.
Supports processing data in chunks to enable job arrays.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from compleximeter import ComplexiMeter

def load_switchboard_speaker_data():
    """Load switchboard speaker level dataset."""
    print("Loading switchboard speaker level dataset...")
    
    df = pd.read_csv('data/switchboard_speaker_level.csv')
    print(f"Switchboard dataset: {df.shape[0]} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df

def convert_to_compleximeter_dataset(df, chunk_id, total_chunks):
    """Convert switchboard dataframe to compleximeter dataset format."""
    print(f"Converting to compleximeter dataset format for chunk {chunk_id}/{total_chunks}...")
    
    dataset = []
    total_rows = len(df)
    
    chunk_size = total_rows // total_chunks
    start_idx = (chunk_id - 1) * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < total_chunks else total_rows
    
    print(f"Processing rows {start_idx} to {end_idx} (chunk {chunk_id}/{total_chunks})")
    
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        
        if (idx - start_idx + 1) % 100 == 0:
            print(f"Progress: Processed {idx - start_idx + 1}/{end_idx - start_idx} rows in chunk", flush=True)
        
        # Process speaker A text
        a_text = row['A_text']
        if pd.notna(a_text) and str(a_text).strip():
            dataset.append({
                'id': f"row_{idx}_speaker_A",
                'text': str(a_text).strip(),
                'speaker': 'A',
                'row_id': idx
            })
        
        # Process speaker B text
        b_text = row['B_text']
        if pd.notna(b_text) and str(b_text).strip():
            dataset.append({
                'id': f"row_{idx}_speaker_B",
                'text': str(b_text).strip(),
                'speaker': 'B',
                'row_id': idx
            })
    
    print(f"Converted to dataset with {len(dataset)} total texts for chunk {chunk_id}", flush=True)
    return dataset

def main():
    """Process switchboard complexity data using compleximeter."""
    # Get chunk information from SLURM environment variables
    chunk_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    total_chunks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 10))
    
    print(f"Starting switchboard complexity data processing for chunk {chunk_id}/{total_chunks}...")
    print(f"SLURM_ARRAY_TASK_ID: {chunk_id}")
    print(f"SLURM_ARRAY_TASK_COUNT: {total_chunks}")
    
    try:
        df = load_switchboard_speaker_data()
        
        dataset = convert_to_compleximeter_dataset(df, chunk_id, total_chunks)
        
        if not dataset:
            print("No data to process for this chunk")
            return
        
        print("Initializing ComplexiMeter...")
        cm = ComplexiMeter()
        
        print("Processing dataset with compleximeter...")
        output_file = f'complexity_results_switchboard/complexity_results_chunk_{chunk_id}.csv'
        
        # Create output directory if it doesn't exist
        os.makedirs('complexity_results_switchboard', exist_ok=True)
        
        cm.process_dataset(dataset, output_file)
        
        print("Loading results and adding speaker information...")
        results_df = pd.read_csv(output_file)
        
        # Extract speaker and row_id from the id column
        results_df['speaker'] = results_df['id'].str.extract(r'speaker_([AB])')
        results_df['row_id'] = results_df['id'].str.extract(r'row_(\d+)_speaker').astype(int)
        
        results_df = results_df.rename(columns={'input': 'text'})
        
        # Reorder columns to have text, speaker, row_id first
        cols = ['text', 'speaker', 'row_id'] + [col for col in results_df.columns 
                                              if col not in ['text', 'speaker', 'row_id', 'id']]
        results_df = results_df[cols]
        
        final_output = f'complexity_results_switchboard/complexity_results_chunk_{chunk_id}_with_speakers.csv'
        results_df.to_csv(final_output, index=False, encoding='utf-8')
        
        print(f"\nSummary for chunk {chunk_id}:")
        print(f"Processed texts: {len(results_df)}", flush=True)
        print(f"Speakers present: {sorted(results_df['speaker'].unique())}", flush=True)
        print(f"Results saved to: {final_output}", flush=True)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 