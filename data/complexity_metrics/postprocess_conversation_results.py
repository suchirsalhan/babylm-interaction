#!/usr/bin/env python3
"""
Postprocessing script to add missing metadata columns to existing conversation complexity results.
This script loads the conversation data and matches it with the existing CSV to add complexity and topic columns.
"""

import pandas as pd
import json
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

def extract_metadata_from_conversations(conversations):
    """Extract metadata from conversations."""
    print("Extracting metadata for utterances...")
    
    metadata_map = {}
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
                
                utterance_id = f"conv_{conv_idx}_utt_{utt_idx}"
                
                metadata_map[utterance_id] = {
                    'complexity': complexity,
                    'topic': topic,
                    'conversation_id': conv_idx,
                    'utterance_id': utt_idx
                }
    
    print(f"Extracted metadata for {len(metadata_map)} utterances", flush=True)
    return metadata_map

def postprocess_individual_utterances():
    """Postprocess individual utterances."""
    print("Postprocessing individual utterances CSV...")
    
    try:
        results_df = pd.read_csv('individual_utterances_complexity.csv')
        print(f"Loaded CSV with {len(results_df)} rows")
    except FileNotFoundError:
        print("ERROR: Could not find 'individual_utterances_complexity.csv'")
        sys.exit(1)
    
    conversations = load_conversation_data()
    metadata_map = extract_metadata_from_conversations(conversations)
    
    print("Adding metadata columns...")
    results_df['complexity'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('complexity', 'unknown'))
    results_df['topic'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('topic', 'unknown'))
    results_df['conversation_id'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('conversation_id', -1))
    results_df['utterance_id'] = results_df['id'].map(lambda x: metadata_map.get(x, {}).get('utterance_id', -1))
    
    if 'input' in results_df.columns:
        results_df = results_df.rename(columns={'input': 'text'})
    
    cols = ['text', 'complexity', 'topic', 'conversation_id', 'utterance_id'] + [col for col in results_df.columns 
                                          if col not in ['text', 'complexity', 'topic', 'conversation_id', 'utterance_id', 'id']]
    results_df = results_df[cols]
    
    final_output = 'individual_utterances_complexity_results.csv'
    results_df.to_csv(final_output, index=False, encoding='utf-8')
    
    print(f"Postprocessed results saved to: {final_output}")
    
    print("\nSummary:")
    print(f"Total utterances processed: {len(results_df)}")
    print(f"Complexity levels found: {results_df['complexity'].value_counts().to_dict()}")
    print(f"Unique topics: {results_df['topic'].nunique()}")
    
    return results_df

def postprocess_dialog_aggregations():
    """Postprocess dialog aggregations."""
    try:
        dialog_df = pd.read_csv('dialog_aggregations_complexity.csv')
        print(f"Found dialog aggregations CSV with {len(dialog_df)} rows")
        
        conversations = load_conversation_data()
        
        metadata_map = {}
        for conv_idx, conversation in enumerate(conversations):
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
                aggregation_id = f"conv_{conv_idx}_aggregated"
                metadata_map[aggregation_id] = {
                    'complexity': complexity,
                    'topic': topic,
                    'conversation_id': conv_idx,
                    'utterance_count': len(instructor_utterances)
                }
        
        dialog_df['complexity'] = dialog_df['id'].map(lambda x: metadata_map.get(x, {}).get('complexity', 'unknown'))
        dialog_df['topic'] = dialog_df['id'].map(lambda x: metadata_map.get(x, {}).get('topic', 'unknown'))
        dialog_df['conversation_id'] = dialog_df['id'].map(lambda x: metadata_map.get(x, {}).get('conversation_id', -1))
        dialog_df['utterance_count'] = dialog_df['id'].map(lambda x: metadata_map.get(x, {}).get('utterance_count', 0))
        
        if 'input' in dialog_df.columns:
            dialog_df = dialog_df.rename(columns={'input': 'text'})
        
        cols = ['text', 'complexity', 'topic', 'conversation_id', 'utterance_count'] + [col for col in dialog_df.columns 
                                          if col not in ['text', 'complexity', 'topic', 'conversation_id', 'utterance_count', 'id']]
        dialog_df = dialog_df[cols]
        
        final_output = 'dialog_aggregations_complexity_results.csv'
        dialog_df.to_csv(final_output, index=False, encoding='utf-8')
        
        print(f"Dialog aggregations postprocessed and saved to: {final_output}")
        return dialog_df
        
    except FileNotFoundError:
        print("No dialog aggregations CSV found, skipping...")
        return None

def main():
    """Postprocess conversation results."""
    print("Starting postprocessing of conversation complexity results...")
    
    try:
        print("\n" + "="*50)
        print("POSTPROCESSING INDIVIDUAL UTTERANCES")
        print("="*50)
        
        individual_results = postprocess_individual_utterances()
        
        print("\n" + "="*50)
        print("POSTPROCESSING DIALOG AGGREGATIONS")
        print("="*50)
        
        dialog_results = postprocess_dialog_aggregations()
        
        print("\n" + "="*50)
        print("POSTPROCESSING COMPLETE")
        print("="*50)
        print("Files created:")
        print("- individual_utterances_complexity_results.csv")
        if dialog_results is not None:
            print("- dialog_aggregations_complexity_results.csv")
        
        print("\nSample individual utterances:")
        print(individual_results[['text', 'complexity', 'topic']].head(10))
        
        if dialog_results is not None:
            print("\nSample dialog aggregations:")
            print(dialog_results[['text', 'complexity', 'topic', 'utterance_count']].head(5))
        
    except Exception as e:
        print(f"Error during postprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 