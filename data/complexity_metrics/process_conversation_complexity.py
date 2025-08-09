#!/usr/bin/env python3
"""
Script to process conversation complexity data using the existing process_dataset method from compleximeter.
Extracts instructor utterances from conversation analysis data and calculates complexity metrics for:
1. Each individual instructor utterance
2. Instructor utterances aggregated per dialog

The results are returned in two separate CSV files with "complexity" as level for data from 5levels_conversation_analysis.
"""

import pandas as pd
import numpy as np
import json
from compleximeter import ComplexiMeter
import os
import sys

def load_conversation_data():
    """Load conversation data."""
    print("Loading conversation analysis data...")
    
    try:
        with open('data/5levels_conversation_analysis.json', 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"Loaded {len(conversations)} conversations")
        return conversations
    except FileNotFoundError:
        print("ERROR: Could not find 'data/5levels_conversation_analysis.json'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in conversation data: {e}")
        sys.exit(1)

def extract_instructor_utterances(conversations):
    """Extract instructor utterances."""
    print("Extracting individual instructor utterances...")
    
    individual_utterances = []
    total_conversations = len(conversations)
    
    for conv_idx, conversation in enumerate(conversations):
        if (conv_idx + 1) % 100 == 0:
            print(f"Progress: Processed {conv_idx + 1}/{total_conversations} conversations ({(conv_idx + 1)/total_conversations*100:.1f}%)", flush=True)
        
        topic = conversation.get('topic', f'conversation_{conv_idx}')
        complexity = conversation.get('complexity', 'unknown')
        dialogue_sequence = conversation.get('dialogue_sequence', [])
        
        for utt_idx, utterance_data in enumerate(dialogue_sequence):
            speaker = utterance_data.get('speaker', '')
            utterance_text = utterance_data.get('utterance', '')
            
            if speaker.lower() == 'instructor' and utterance_text.strip():
                clean_text = utterance_text.strip()
                if clean_text.startswith('- '):
                    clean_text = clean_text[2:]
                
                individual_utterances.append({
                    'id': f"conv_{conv_idx}_utt_{utt_idx}",
                    'text': clean_text,
                    'complexity': complexity,
                    'topic': topic,
                    'conversation_id': conv_idx,
                    'utterance_id': utt_idx
                })
    
    print(f"Extracted {len(individual_utterances)} individual instructor utterances", flush=True)
    return individual_utterances

def aggregate_utterances_per_dialog(conversations):
    """Aggregate utterances per dialog."""
    print("Aggregating instructor utterances per dialog...")
    
    dialog_aggregations = []
    total_conversations = len(conversations)
    
    for conv_idx, conversation in enumerate(conversations):
        if (conv_idx + 1) % 100 == 0:
            print(f"Progress: Processed {conv_idx + 1}/{total_conversations} conversations ({(conv_idx + 1)/total_conversations*100:.1f}%)", flush=True)
        
        topic = conversation.get('topic', f'conversation_{conv_idx}')
        complexity = conversation.get('complexity', 'unknown')
        dialogue_sequence = conversation.get('dialogue_sequence', [])
        
        instructor_utterances = []
        for utterance_data in dialogue_sequence:
            speaker = utterance_data.get('speaker', '')
            utterance_text = utterance_data.get('utterance', '')
            
            if speaker.lower() == 'instructor' and utterance_text.strip():
                clean_text = utterance_text.strip()
                if clean_text.startswith('- '):
                    clean_text = clean_text[2:]
                instructor_utterances.append(clean_text)
        
        if instructor_utterances:
            aggregated_text = ' '.join(instructor_utterances)
            
            dialog_aggregations.append({
                'id': f"conv_{conv_idx}_aggregated",
                'text': aggregated_text,
                'complexity': complexity,
                'topic': topic,
                'conversation_id': conv_idx,
                'utterance_count': len(instructor_utterances)
            })
    
    print(f"Created {len(dialog_aggregations)} dialog aggregations", flush=True)
    return dialog_aggregations

def process_individual_utterances(utterances):
    """Process individual utterances."""
    print("Processing individual instructor utterances...")
    
    print("Initializing ComplexiMeter...")
    cm = ComplexiMeter()
    
    print("Processing individual utterances with compleximeter...")
    output_file = 'individual_utterances_complexity.csv'
    cm.process_dataset(utterances, output_file)
    
    print("Loading results and adding metadata...")
    results_df = pd.read_csv(output_file)
    
    results_df['conversation_id'] = results_df['id'].str.extract(r'conv_(\d+)_utt').astype(int)
    results_df['utterance_id'] = results_df['id'].str.extract(r'utt_(\d+)').astype(int)
    
    results_df = results_df.rename(columns={'input': 'text'})
    
    metadata_map = {}
    for utterance in utterances:
        metadata_map[utterance['id']] = {
            'complexity': utterance.get('complexity', 'unknown'),
            'topic': utterance.get('topic', 'unknown')
        }
    
    results_df['complexity'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('complexity', 'unknown'))
    results_df['topic'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('topic', 'unknown'))
    
    cols = ['text', 'complexity', 'topic', 'conversation_id', 'utterance_id'] + [col for col in results_df.columns 
                                          if col not in ['text', 'complexity', 'topic', 'conversation_id', 'utterance_id', 'id']]
    results_df = results_df[cols]
    
    final_output = 'individual_utterances_complexity_results.csv'
    results_df.to_csv(final_output, index=False, encoding='utf-8')
    
    print(f"Individual utterances results saved to: {final_output}")
    return results_df

def process_dialog_aggregations(aggregations):
    """Process dialog aggregations."""
    print("Processing dialog aggregations...")
    
    print("Initializing ComplexiMeter...")
    cm = ComplexiMeter()
    
    print("Processing dialog aggregations with compleximeter...")
    output_file = 'dialog_aggregations_complexity.csv'
    cm.process_dataset(aggregations, output_file)
    
    print("Loading results and adding metadata...")
    results_df = pd.read_csv(output_file)
    
    results_df['conversation_id'] = results_df['id'].str.extract(r'conv_(\d+)_aggregated').astype(int)
    
    results_df = results_df.rename(columns={'input': 'text'})
    
    metadata_map = {}
    for aggregation in aggregations:
        metadata_map[aggregation['id']] = {
            'complexity': aggregation.get('complexity', 'unknown'),
            'topic': aggregation.get('topic', 'unknown'),
            'utterance_count': aggregation.get('utterance_count', 0)
        }
    
    results_df['complexity'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('complexity', 'unknown'))
    results_df['topic'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('topic', 'unknown'))
    results_df['utterance_count'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('utterance_count', 0))
    
    cols = ['text', 'complexity', 'topic', 'conversation_id', 'utterance_count'] + [col for col in results_df.columns 
                                          if col not in ['text', 'complexity', 'topic', 'conversation_id', 'utterance_count', 'id']]
    results_df = results_df[cols]
    
    final_output = 'dialog_aggregations_complexity_results.csv'
    results_df.to_csv(final_output, index=False, encoding='utf-8')
    
    print(f"Dialog aggregations results saved to: {final_output}")
    return results_df

def main():
    """Process conversation complexity data."""
    print("Starting conversation complexity data processing...")
    
    try:
        conversations = load_conversation_data()
        
        print("\n" + "="*50)
        print("PROCESSING INDIVIDUAL INSTRUCTOR UTTERANCES")
        print("="*50)
        
        #individual_utterances = extract_instructor_utterances(conversations)
       # individual_results = process_individual_utterances(individual_utterances)
        
        print("\n" + "="*50)
        print("PROCESSING DIALOG AGGREGATIONS")
        print("="*50)
        
        dialog_aggregations = aggregate_utterances_per_dialog(conversations)
        dialog_results = process_dialog_aggregations(dialog_aggregations)
        
        
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 