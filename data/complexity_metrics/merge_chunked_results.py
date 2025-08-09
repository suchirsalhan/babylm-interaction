#!/usr/bin/env python3
"""
Script to merge results from chunked complexity analysis jobs.
"""

import pandas as pd
import glob
import os
import sys

def merge_chunked_results(total_chunks=10):
    """Merge chunked results."""
    print("Merging chunked complexity analysis results...")
    
    print("Merging results with levels...")
    all_results_with_levels = []
    
    for chunk_id in range(1, total_chunks + 1):
        chunk_file = f"complexity_results_with_levels/complexity_results_chunk_{chunk_id}_with_levels.csv"
        
        if os.path.exists(chunk_file):
            print(f"Loading chunk {chunk_id} from {chunk_file}")
            chunk_df = pd.read_csv(chunk_file)
            all_results_with_levels.append(chunk_df)
            print(f"  - Loaded {len(chunk_df)} rows from chunk {chunk_id}")
        else:
            print(f"Warning: Chunk file {chunk_file} not found")
    
    if all_results_with_levels:
        print("Merging all chunks with levels...")
        merged_with_levels = pd.concat(all_results_with_levels, ignore_index=True)
        
        merged_with_levels = merged_with_levels.sort_values(['row_id', 'level']).reset_index(drop=True)
        
        output_file_with_levels = 'complexity_results_with_levels_merged.csv'
        merged_with_levels.to_csv(output_file_with_levels, index=False, encoding='utf-8')
        
        print(f"\nMerged results with levels saved to: {output_file_with_levels}")
        print(f"Total rows: {len(merged_with_levels)}")
        print(f"Unique row_ids: {merged_with_levels['row_id'].nunique()}")
        print(f"Levels present: {sorted(merged_with_levels['level'].unique())}")
    else:
        print("No chunk files with levels found!")
    
    print("\nMerging results without levels...")
    all_results_without_levels = []
    
    for chunk_id in range(1, total_chunks + 1):
        chunk_file = f"complexity_results_without_levels/complexity_results_chunk_{chunk_id}.csv"
        
        if os.path.exists(chunk_file):
            print(f"Loading chunk {chunk_id} from {chunk_file}")
            chunk_df = pd.read_csv(chunk_file)
            all_results_without_levels.append(chunk_df)
            print(f"  - Loaded {len(chunk_df)} rows from chunk {chunk_id}")
        else:
            print(f"Warning: Chunk file {chunk_file} not found")
    
    if all_results_without_levels:
        print("Merging all chunks without levels...")
        merged_without_levels = pd.concat(all_results_without_levels, ignore_index=True)
        
        merged_without_levels = merged_without_levels.sort_values('id').reset_index(drop=True)
        
        output_file_without_levels = 'complexity_results_without_levels_merged.csv'
        merged_without_levels.to_csv(output_file_without_levels, index=False, encoding='utf-8')
        
        print(f"\nMerged results without levels saved to: {output_file_without_levels}")
        print(f"Total rows: {len(merged_without_levels)}")
    else:
        print("No chunk files without levels found!")
    
    if all_results_with_levels:
        print("\nSample of merged results with levels:")
        print(merged_with_levels[['text', 'level', 'row_id']].head(10))

if __name__ == "__main__":
    total_chunks = 10
    merge_chunked_results(total_chunks) 