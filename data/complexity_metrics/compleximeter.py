import pandas as pd
import spacy
import nltk
import re
import numpy as np
from textstat import flesch_kincaid_grade, coleman_liau_index, smog_index, gunning_fog, syllable_count
import taaled
from nltk.corpus import wordnet, brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import cefrpy
from sentence_concreteness.sentence_concreteness import get_sentence_concreteness
import itertools
import networkx as nx
import requests
import time
import os
from connectives_list import CONNECTIVES

class ConceptNetAPI:
    def __init__(self):
        """Initialize ConceptNet API client."""
        self.base_url = "http://api.conceptnet.io"
        self.session = requests.Session()
    
    def lookup(self, word):
        """Look up a word in ConceptNet API."""
        try:
            response = self.session.get(f"{self.base_url}/c/en/{word}", timeout=2)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

class DBpediaAPI:
    def __init__(self):
        """Initialize DBpedia API client with fallback to local entity recognition."""
        self.spotlight_url = "https://api.dbpedia-spotlight.org/en/annotate"
        self.sparql_url = "https://dbpedia.org/sparql"
        self.session = requests.Session()
        self.network_available = True
        self.local_recognizer = None
        self._test_connectivity()
        self._setup_local_fallback()
    
    def _test_connectivity(self):
        """[Helper function] Test network connectivity to DBpedia APIs."""
        try:
            response = self.session.get(
                self.spotlight_url,
                params={'text': 'test', 'confidence': 0.5},
                headers={'Accept': 'application/json'},
                timeout=2
            )
            self.network_available = True
        except requests.exceptions.RequestException:
            self.network_available = False
            print("WARNING: DBpedia APIs are not accessible. Using local entity recognition as fallback.")
    
    def _setup_local_fallback(self):
        """[Helper function] Set up DBpedia entity recognition as fallback."""
        if not self.network_available:
            try:
                from local_entity_recognition import DBpediaRecognizer
                self.local_recognizer = DBpediaRecognizer(
                    cache_ttl_hours=24,
                    max_retries=3,
                    request_timeout=30,
                    confidence_threshold=0.5
                )
                print("SUCCESS: DBpedia entity recognition system loaded as fallback")
                self._verify_local_entity_recognition()
            except ImportError:
                print("WARNING: DBpedia entity recognition not available. DBpedia metrics will return default values.")
    
    def _verify_local_entity_recognition(self):
        """[Helper function] Verify local entity recognition functionality."""
        try:
            test_text = "Barack Obama attended Harvard University."
            entities = self.local_recognizer.extract_entities(test_text)
            if entities:
                print(f"SUCCESS: Local entity recognition verified: {len(entities)} entities found in test")
            else:
                print("WARNING:  Local entity recognition found but no entities extracted in test")
        except Exception as e:
            print(f"WARNING:  Local entity recognition loaded but verification failed: {e}")
    
    def annotate(self, text, confidence=0.5):
        """Annotate text with DBpedia entities using Spotlight API."""
        if not text:
            return []
        if self.network_available:
            try:
                response = self.session.get(
                    self.spotlight_url,
                    params={'text': text, 'confidence': confidence},
                    headers={'Accept': 'application/json'},
                    timeout=2
                )
                response.raise_for_status()
                data = response.json()
                if 'Resources' in data:
                    uris = [resource['@URI'] for resource in data['Resources']]
                    return uris
                return []
            except requests.exceptions.ConnectionError as e:
                self.network_available = False
                return self._annotate_local(text)
            except requests.exceptions.Timeout as e:
                return self._annotate_local(text)
            except requests.exceptions.RequestException as e:
                if self.network_available:
                    print(f"WARNING: DBpedia  API error: {e}")
                    self.network_available = False
                return self._annotate_local(text)
        return self._annotate_local(text)
    
    def _annotate_local(self, text):
        """[Helper function] Annotate text using DBpedia entity recognition."""
        if self.local_recognizer:
            entity_matches = self.local_recognizer.extract_entities(text)
            return [match.dbpedia_uri for match in entity_matches]
        return []

    def are_related(self, uri1, uri2):
        """Check if two entities are related using DBpedia SPARQL."""
        if self.network_available:
            query = f"""
            ASK {{
                {{ <{uri1}> ?p <{uri2}> }}
                UNION
                {{ <{uri2}> ?p <{uri1}> }}
            }}
            """
            try:
                response = self.session.get(
                    self.sparql_url,
                    params={'query': query, 'format': 'json'},
                    headers={
                        'Accept': 'application/sparql-results+json',
                        'User-Agent': 'Mozilla/5.0 (compatible; DBpediaTest/1.0)'
                    },
                    timeout=3
                )
                response.raise_for_status()
                data = response.json()
                return data.get('boolean', False)
            except requests.exceptions.RequestException:
                pass
        if self.local_recognizer:
            related, confidence = self.local_recognizer.are_entities_related(uri1, uri2)
            return related
        return False

#The object measuring various complexity metrics
class ComplexiMeter:
    def __init__(self, metrics_file='Metrics.csv', crat_path='CRAT_v1.1.app'):
        """Initialize ComplexiMeter with NLP models and databases."""
       
        self.use_spacy = True
        self.nlp = None
        self.analyzer = None

        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Spacy model 'en_core_web_sm' not found.")
                self.nlp = None
            
            if self.nlp:
                self.analyzer = cefrpy.CEFRAnalyzer(self.nlp)
        
        from sentence_concreteness.sentence_concreteness import get_sentence_concreteness
        self.score_concreteness = get_sentence_concreteness
        
        self.conceptnet = ConceptNetAPI()
        self.dbpedia = DBpediaAPI()

        self._download_nltk_data()
        self.graph_cache_conceptnet = {}
        self.graph_cache_dbpedia = {}
        self.session = requests.Session()
        
        try:
            self.metrics_df = pd.read_csv(metrics_file)
        except FileNotFoundError:
            print(f"Metrics file '{metrics_file}' not found.")
            self.metrics_df = pd.DataFrame()

        # Load CRAT databases
        self.crat_path = crat_path
        self.kuperman_aoa = self._load_crat_db('AoA_Brysbart.txt', 1, has_header=True)
        self.mrc_familiarity = self._load_crat_db('MRC_database_simple_final_lower.txt', 2, has_header=False)
        self.mrc_imagery = self._load_crat_db('MRC_database_simple_final_lower.txt', 5, has_header=False)
        self.mrc_meaningfulness = self._load_crat_db('MRC_database_simple_final_lower.txt', 6, has_header=False)
        self.coca_academic_unigram = self._load_crat_db('COCA_academic_unigram_list.csv', 1, has_header=True)
        self.coca_academic_frequency = self._load_crat_db('COCA_acad_word_list_lemma_freq.csv', 1, has_header=False)
        self.word_freq_db = self._load_crat_db('MRC_database_simple_final_lower.txt', 4, has_header=False)

        self.stemmer = PorterStemmer()
        self.taaco_cache = {}

    def _load_crat_db(self, filename, column_index, has_header=True):
        """[Helper function] Load lexicon from CRAT resources."""
        db = {}
        file_path = os.path.join(self.crat_path, 'Contents', 'Resources', filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if has_header:
                    next(f) 
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 1:
                        parts = line.strip().split(',')

                    if len(parts) > column_index:
                        word = parts[0].lower()
                        try:
                            value = float(parts[column_index])
                            db[word] = value
                        except (ValueError, IndexError):
                            continue
        except FileNotFoundError:
            print(f"WARNING: CRAT database file not found at '{file_path}'")
        
        return db

    def _get_concepts_from_text_conceptnet(self, text):
        """[Helper function] Extract concepts from text using spaCy for ConceptNet."""
        if not self.nlp or not text:
            return []
        doc = self.nlp(text)
        concepts = [
            token.lemma_.lower() for token in doc
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop
        ]
        return concepts

    def _get_concepts_from_text_dbpedia(self, text):
        """[Helper function] Extract DBpedia entities from text."""
        if not self.dbpedia or not text:
            return []
        return self.dbpedia.annotate(text)

    def _download_nltk_data(self):
        """[Helper function] Check and download required NLTK data packages."""
        required_data = {
            'tokenizers': ['punkt'],
            'corpora': ['wordnet', 'omw-1.4', 'wordnet_ic', 'brown', 'stopwords'],
            'taggers': ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
        }
        
        missing_data = []
        for data_type, datasets in required_data.items():
            for dataset in datasets:
                try:
                    nltk.data.find(f'{data_type}/{dataset}')
                except LookupError:
                    missing_data.append(f'{data_type}/{dataset}')
        
        if missing_data:
            print("WARNING: The following NLTK data packages are missing:")
            for dataset in missing_data:
                print(f"  - {dataset}")
           

    def _get_relatedness_conceptnet(self, concept1, concept2):
        """[Helper function] Query ConceptNet API for relatedness between two concepts."""
        url = f"http://api.conceptnet.io/relatedness?node1=/c/en/{concept1}&node2=/c/en/{concept2}"
        try:
            time.sleep(0.05)
            response = self.session.get(url, timeout=3)
            response.raise_for_status()
            return response.json().get('value', 0)
        except requests.exceptions.RequestException as e:
            return -1

    def _build_text_graph_conceptnet(self, text, threshold=0.05):
        """[Helper function] Build knowledge graph from text using ConceptNet."""
        text_hash = hash(text)
        if text_hash in self.graph_cache_conceptnet:
            return self.graph_cache_conceptnet[text_hash]

        if not self.nlp:
            return nx.Graph()

        doc = self.nlp(text)
        G = nx.Graph()
        
        concepts = set(
            token.lemma_.lower() for token in doc 
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop
        )
        G.add_nodes_from(concepts)

        checked_pairs = set()

        for token in doc:
            if token.is_stop or token.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                continue

            head = token.head
            if head.is_stop or head.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ'] or head == token:
                pass
            else:
                concept1 = token.lemma_.lower()
                concept2 = head.lemma_.lower()

                pair = tuple(sorted((concept1, concept2)))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    relatedness = self._get_relatedness_conceptnet(concept1, concept2)
                    if relatedness > threshold:   
                        G.add_edge(concept1, concept2, weight=relatedness)

            for sibling in token.head.children:
                if sibling == token or sibling.is_stop or sibling.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                    continue
            
                concept1 = token.lemma_.lower()
                concept2 = sibling.lemma_.lower()
                pair = tuple(sorted((concept1, concept2)))

                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    relatedness = self._get_relatedness_conceptnet(concept1, concept2)
                    if relatedness > threshold:  
                        G.add_edge(concept1, concept2, weight=relatedness)
        
        self.graph_cache_conceptnet[text_hash] = G
        return G

    def _build_text_graph_dbpedia(self, text):
        """
        Builds a knowledge graph from the text by linking entities using DBpedia.
        1. Extracts named entities from the text using DBpedia Spotlight.
        2. For each pair of unique entities, queries DBpedia to see if they are directly linked.
        3. Creates a graph with entities as nodes and DBpedia links as edges.
        """
        text_hash = hash(text)
        if text_hash in self.graph_cache_dbpedia:
            return self.graph_cache_dbpedia[text_hash]

        entities = self._get_concepts_from_text_dbpedia(text)
        if not entities:
            G = nx.Graph()
            self.graph_cache_dbpedia[text_hash] = G
            return G
        
        unique_entities = sorted(list(set(entities)))
        
        G = nx.Graph()
        G.add_nodes_from(unique_entities)

        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                uri1 = unique_entities[i]
                uri2 = unique_entities[j]
                if self.dbpedia.are_related(uri1, uri2):
                    G.add_edge(uri1, uri2, weight=1.0) 
        
        self.graph_cache_dbpedia[text_hash] = G
        return G

    def _clean_metric_name(self, metric_name):
        """[Helper function] Clean metric name for function mapping."""
        metric_name = re.sub(r'\[.*?\]', '', metric_name)
        metric_name = metric_name.lower()
        metric_name = re.sub(r'[^a-z0-9_ ]', '', metric_name)
        return metric_name.strip().replace(' ', '_')

    # Lexical Metrics
    def type_token_ratio(self, text):
        """Calculate type-token ratio using TAACO."""
        if not text:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        return taaled.lexdiv(tokens).ttr

    def academic_word_list_coverage(self, text):
        """Calculate percentage of academic words using COCA list."""
        if not text or not self.coca_academic_unigram:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        
        academic_words = [token.lower() for token in tokens if token.lower() in self.coca_academic_unigram]
        return len(academic_words) / len(tokens) * 100 if tokens else 0

    def percentage_of_words_above_b1_level(self, text):
        """Calculate percentage of words above B1 CEFR level."""
        if not self.nlp or not text:
            return -1
        try:
            from cefrpy import CEFRSpaCyAnalyzer
            doc = self.nlp(text)
            text_analyzer = CEFRSpaCyAnalyzer()
            tokens_with_levels = text_analyzer.analize_doc(doc)

            above_b1_count = 0
            total_words = 0
            for token in tokens_with_levels:
                if not token[2] and token[3] is not None:
                    total_words += 1
                    if token[3] > 3.0:
                        above_b1_count += 1
            return above_b1_count / total_words * 100 if total_words > 0 else 0
        except Exception as e:
            print(f"Error in percentage_of_words_above_b1_level: {e}")
            return -1

    def average_cefr_level(self, text):
        """Calculate average CEFR level of words in text."""
        if not self.nlp or not text:
            return -1
        
        try:
            from cefrpy import CEFRSpaCyAnalyzer
            

            text_analyzer = CEFRSpaCyAnalyzer()
            doc = self.nlp(text)
            
            tokens_with_levels = text_analyzer.analize_doc(doc)
            
            levels = [
                token[3] for token in tokens_with_levels 
                if not token[2] and token[3] is not None
            ]
            
            if not levels:
                return -1  
            
            return sum(levels) / len(levels)
            
        except Exception as e:
            print(f"Error calculating average_cefr_level: {e}")
            return -1
    
    def average_number_of_meaning_per_word(self, text):
        """Calculate average number of WordNet synsets per word."""
        if not text:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
            
        total_meanings = 0
        word_count = 0
        for token in tokens:
            if token.isalpha() and token.lower() not in self.nlp.Defaults.stop_words:
                total_meanings += len(wordnet.synsets(token))
                word_count += 1
            
        return total_meanings / word_count if word_count > 0 else 0

    def percentage_of_words_with_more_than_5_meanings(self, text):
        """Calculate percentage of words with more than 5 WordNet meanings."""
        if not text:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        
        count = 0
        word_count = 0
        for token in tokens:
            if token.isalpha() and token.lower() not in self.nlp.Defaults.stop_words:
                if len(wordnet.synsets(token)) > 5:
                    count += 1
                word_count += 1
        
        return count / word_count * 100 if word_count > 0 else 0

    def _get_psycholinguistic_scores(self, text, db):
        """[Helper function] Extract psycholinguistic scores from text using database."""
        if not text or not db or not self.nlp:
            return []
        doc = self.nlp(text)
        scores = []
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in db:
                value = db[lemma]
                try:
                    fval = float(value)
                    if db is self.kuperman_aoa:
                        if 1 <= fval <= 20:
                            scores.append(fval)
                        else:
                            print(f"[AoA DEBUG] Out-of-range value for '{lemma}': {fval}")
                    else:
                        scores.append(fval)
                except Exception as e:
                    if db is self.kuperman_aoa:
                        print(f"[AoA DEBUG] Non-float value for '{lemma}': {value} ({e})")
        return scores

    def min_kuperman_age_of_acquisition(self, text):
        """Calculate minimum Kuperman age of acquisition score."""
        scores = self._get_psycholinguistic_scores(text, self.kuperman_aoa)
        return min(scores) if scores else 0

    def max_kuperman_age_of_acquisition(self, text):
        """Calculate maximum Kuperman age of acquisition score."""
        scores = self._get_psycholinguistic_scores(text, self.kuperman_aoa)
        return max(scores) if scores else 0
    
    def median_kuperman_age_of_acquisition(self, text):
        """Calculate median Kuperman age of acquisition score."""
        scores = self._get_psycholinguistic_scores(text, self.kuperman_aoa)
        return np.median(scores) if scores else 0

    def min_mrc_familiarity(self, text):
        """Calculate minimum MRC familiarity score."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_familiarity)
        return min(scores) if scores else 0

    def max_mrc_familiarity(self, text):
        """Calculate maximum MRC familiarity score."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_familiarity)
        return max(scores) if scores else 0

    def median_mrc_familiarity(self, text):
        """Calculate median MRC familiarity score."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_familiarity)
        return np.median(scores) if scores else 0

    def coca_academic_range(self, text):
        """Calculate average COCA academic frequency score."""
        if not text or not self.nlp:
            return -1
        doc = self.nlp(text)
        scores = [self.coca_academic_unigram.get(token.lemma_.lower(), 0) for token in doc if token.lemma_.lower() in self.coca_academic_unigram]
        return np.mean(scores) if scores else 0
    
    def mean_word_length(self, text):
        """Calculate average word length in characters."""
        if not text:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        return sum(len(word) for word in tokens) / len(tokens)
    
    def number_of_wordnet_hypernyms_per_word(self, text):
        """Calculate average number of WordNet hypernyms per word."""
        if not text:
            return -1
        
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        
        total_hypernyms = 0
        word_count = 0
        
        for token in tokens:
            # Only count alphabetic tokens
            if token.isalpha():
                word_count += 1
                for synset in wordnet.synsets(token):
                    total_hypernyms += len(synset.hypernyms())
        
        return total_hypernyms / word_count if word_count > 0 else 0
        
    def number_of_wordnet_hyponyms_per_word(self, text):
        """Calculate average number of WordNet hyponyms per word."""
        if not text:
            return -1
        
        tokens = word_tokenize(text)
        if not tokens:
            return -1
            
        total_hyponyms = 0
        word_count = 0
        
        for token in tokens:
            # Only count alphabetic tokens
            if token.isalpha():
                word_count += 1
                for synset in wordnet.synsets(token):
                    total_hyponyms += len(synset.hyponyms())
        
        return total_hyponyms / word_count if word_count > 0 else 0

    # Structural Metrics
    def average_sentence_length(self, text):
        """Calculate average number of words per sentence."""
        if not text:
            return -1
        sentences = sent_tokenize(text)
        if not sentences:
            return -1
        words = word_tokenize(text)
        return len(words) / len(sentences)

    def max_number_of_if_per_sentence(self, text):
        """Calculate maximum number of 'if' words in any sentence."""
        if not text:
            return -1
        sentences = sent_tokenize(text)
        if not sentences:
            return -1
        
        max_ifs = 0
        for sentence in sentences:
            if_count = len(re.findall(r'\bif\b', sentence.lower()))
            if if_count > max_ifs:
                max_ifs = if_count
        return max_ifs

    def max_number_of_wh_clauses_per_sentence(self, text):
        """Calculate maximum number of WH-words in any sentence."""
        if not text:
            return -1
        sentences = sent_tokenize(text)
        if not sentences:
            return -1

        wh_words = r'\b(who|what|where|when|why|which)\b'
        max_wh = 0
        for sentence in sentences:
            count = len(re.findall(wh_words, sentence.lower()))
            if count > max_wh:
                max_wh = count
        return max_wh
    
    def number_of_connectives_per_3_sentence_sliding_window(self, text):
        """Calculate maximum connectives in 3-sentence sliding window."""
        if not text:
            return -1
        
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            basic_connectives = CONNECTIVES['all_connective']
            text_lower = text.lower()
            count = 0
            for conn in basic_connectives:
                pattern = r'\b' + re.escape(conn) + r'\b'
                count += len(re.findall(pattern, text_lower))
            return count / len(sentences) if sentences else 0

        #set of all connectives
        all_connectives = set()
        for sublist in CONNECTIVES.values():
            all_connectives.update(sublist)
        
        max_connectives = 0
        
        for i in range(len(sentences) - 2):
            window = " ".join(sentences[i:i+3])
            text_lower = window.lower()
            count = 0
            for conn in all_connectives:
                pattern = r'\b' + re.escape(conn) + r'\b'
                count += len(re.findall(pattern, text_lower))
            if count > max_connectives:
                max_connectives = count
            
        return max_connectives

    def number_of_connectives(self, text):
        """Calculate total number of connectives in text."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['all_connective']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def average_number_of_commas_per_sentence(self, text):
        """Calculate average number of commas per sentence."""
        if not text:
            return -1
        sentences = sent_tokenize(text)
        if not sentences:
            return -1
        comma_count = text.count(',')
        return comma_count / len(sentences)

    def flesch_kincaid_grade_level(self, text):
        """Calculate Flesch-Kincaid grade level."""
        if not text or len(word_tokenize(text))<100:
            return -1
        try:
            return flesch_kincaid_grade(text)
        except Exception as e:
            # Fallback to manual calculation for problematic words
            try:
                return self._calculate_flesch_kincaid_manual(text)
            except Exception as e2:
                print(f"WARNING: Manual Flesch-Kincaid calculation also failed: {e2}", flush=True)
                return -1

    def coleman_liau_index(self, text):
        """Calculate Coleman-Liau index."""
        if not text or len(word_tokenize(text))<300:
            return -1
        try:
            return coleman_liau_index(text)
        except Exception as e:
            # Fallback to manual calculation for problematic words
            try:
                return self._calculate_coleman_liau_manual(text)
            except Exception as e2:
                print(f"WARNING: Manual Coleman-Liau calculation also failed: {e2}", flush=True)
                return -1
    
    def smog_index(self, text):
        """Calculate SMOG index."""
        if not text:
            return -1
        sentences = sent_tokenize(text)
        if len(sentences) < 30:
             return -1
        try:
            return smog_index(text)
        except Exception as e:
            # Fallback to manual calculation for problematic words
            try:
                return self._calculate_smog_manual(text)
            except Exception as e2:
                print(f"WARNING: Manual SMOG calculation also failed: {e2}", flush=True)
                return -1

    def gunning_fog_index(self, text):
        """Calculate Gunning Fog index using manual calculation only."""
        if not text:
            return -1
        try:
            return self._calculate_gunning_fog_manual(text)
        except Exception as e:
            print(f"WARNING: Manual Gunning Fog calculation failed: {e}", flush=True)
            return -1

        """if not text:
            return -1
        try:
            return gunning_fog(text)
        except Exception as e:
            # Try to calculate manually by excluding problematic words
            try:
                print(f"WARNING: Error with textstat gunning_fog ({e}), attempting manual calculation...", flush=True)
                return self._calculate_gunning_fog_manual(text)
            except Exception as e2:
                print(f"WARNING: Manual Gunning Fog calculation also failed: {e2}", flush=True)
                return -1""" 
    def _calculate_gunning_fog_manual(self, text):
        """[Helper function] Manual calculation of Gunning Fog index, excluding words that cause errors."""
        import re
        from textstat import syllable_count
        
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return -1
        
        words = word_tokenize(text)
        if len(words) == 0:
            return -1
        
        complex_words = 0
        valid_words = 0
        
        for word in words:
            if word.isalpha():  
                try:
                    syllables = syllable_count(word)
                    valid_words += 1
                    if syllables >= 3:
                        complex_words += 1
                except Exception:
                    continue
        
        if valid_words == 0:
            return -1
        
        # Calculate Gunning Fog manually
        avg_sentence_length = valid_words / len(sentences)
        percent_complex = (complex_words / valid_words) * 100
        
        gunning_fog_score = 0.4 * (avg_sentence_length + percent_complex)
        return gunning_fog_score

    def _calculate_flesch_kincaid_manual(self, text):
        """[Helper function] Manual calculation of Flesch-Kincaid grade level, excluding words that cause errors."""
        import re
        from textstat import syllable_count
        
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return -1
        
        words = word_tokenize(text)
        if len(words) == 0:
            return -1
        
        total_syllables = 0
        valid_words = 0
        
        for word in words:
            if word.isalpha():  
                try:
                    syllables = syllable_count(word)
                    total_syllables += syllables
                    valid_words += 1
                except Exception:
                    continue
        
        if valid_words == 0:
            return -1
        
        # Calculate Flesch-Kincaid manually
        avg_sentence_length = valid_words / len(sentences)
        avg_syllables_per_word = total_syllables / valid_words
        
        # Flesch-Kincaid Grade Level formula: 0.39 × (total words ÷ total sentences) + 11.8 × (total syllables ÷ total words) - 15.59
        flesch_kincaid_score = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        return flesch_kincaid_score

    def _calculate_coleman_liau_manual(self, text):
        """[Helper function] Manual calculation of Coleman-Liau index, excluding words that cause errors."""
        import re
        from textstat import syllable_count
        
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return -1
        
        words = word_tokenize(text)
        if len(words) == 0:
            return -1
        
        total_letters = 0
        valid_words = 0
        
        for word in words:
            if word.isalpha():  
                total_letters += len(word)
                valid_words += 1
        
        if valid_words == 0:
            return -1
        
        # Calculate Coleman-Liau manually
        avg_sentence_length = valid_words / len(sentences)
        avg_letters_per_word = total_letters / valid_words
        
        # Coleman-Liau Index formula: 0.0588 × (average letters per word × 100) - 0.296 × (average sentences per 100 words) - 15.8
        coleman_liau_score = 0.0588 * (avg_letters_per_word * 100) - 0.296 * (avg_sentence_length * 100) - 15.8
        return coleman_liau_score

    def _calculate_smog_manual(self, text):
        """[Helper function] Manual calculation of SMOG index, excluding words that cause errors."""
        import re
        from textstat import syllable_count
        
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return -1
        
        words = word_tokenize(text)
        if len(words) == 0:
            return -1
        
        polysyllable_count = 0
        valid_words = 0
        
        for word in words:
            if word.isalpha():  
                try:
                    syllables = syllable_count(word)
                    valid_words += 1
                    if syllables >= 3:
                        polysyllable_count += 1
                except Exception:
                    continue
        
        if valid_words == 0:
            return -1
        
        # Calculate SMOG manually
        # SMOG formula: 1.043 × √(number of polysyllables × 30 ÷ number of sentences) + 3.1291
        smog_score = 1.043 * (polysyllable_count * 30 / len(sentences)) ** 0.5 + 3.1291
        return smog_score

    def _count_clauses_in_sentence(self, sentence_doc):
        """[Helper function] Count clauses using spaCy's Universal Dependencies framework."""
        if not sentence_doc or len(sentence_doc) == 0:
            return 1  
        
        clause_count = 0
        
        for token in sentence_doc:
            
            if token.dep_ == 'ROOT':
                clause_count += 1
            
            elif token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']:
                clause_count += 1
            
            elif token.dep_ == 'conj' and token.pos_ == 'VERB':
                clause_count += 1
            
            elif token.dep_ in ['csubj', 'csubjpass']:
                clause_count += 1
        
        return max(1, clause_count)

    def min_number_of_clauses_per_sentence(self, text):
        """Calculate minimum number of clauses in any sentence."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        sents = list(doc.sents)
        if not sents:
            return -1
        clause_counts = [self._count_clauses_in_sentence(sent) for sent in sents]
        return min(clause_counts) if clause_counts else 0

    def max_number_of_clauses_per_sentence(self, text):
        """Calculate maximum number of clauses in any sentence."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        sents = list(doc.sents)
        if not sents:
            return -1
        clause_counts = [self._count_clauses_in_sentence(sent) for sent in sents]
        return max(clause_counts) if clause_counts else 0
        
    def average_number_of_clauses_per_sentence(self, text):
        """Calculate average number of clauses per sentence."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        sents = list(doc.sents)
        if not sents:
            return -1
        clause_counts = [self._count_clauses_in_sentence(sent) for sent in sents]
        return np.mean(clause_counts) if clause_counts else 0

    def _get_tree_depth(self, root):
        """[Helper function] Calculate depth of dependency tree."""
        if not list(root.children):
            return 0
        return 1 + max(self._get_tree_depth(child) for child in root.children)

    def dependency_parser_tree_depth(self, text):
        """Calculate average dependency tree depth."""
        if not self.nlp or not text:
            return -1
        
        doc = self.nlp(text)
        if not doc or not doc.sents:
            return -1
        
        tree_depths = []
        for sent in doc.sents:
            root = next((token for token in sent if token.head == token), None)
            if root:
                tree_depths.append(self._get_tree_depth(root))
        
        return sum(tree_depths) / len(tree_depths) if tree_depths else 0

    def dependency_parser_dependency_distance(self, text):
        """Calculate average dependency distance between words and their heads."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        if not doc:
            return -1
        
        total_distance = 0
        token_count = 0
        for token in doc:
            if token.head != token: 
                total_distance += abs(token.i - token.head.i)
                token_count += 1
        
        return total_distance / token_count if token_count > 0 else 0

    def dependency_parser_branching(self, text):
        """Calculate average number of children per token in dependency tree."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        tokens = [token for token in doc]
        if not tokens:
            return -1
        
        total_children = sum(len(list(token.children)) for token in doc)
        return total_children / len(tokens)

    def narrativity_cox(self, text):
        """Returns the narrativity score for a single text."""
        if not text:
            return -1
        
        try:
            if len(text.split()) < 10:
                return -1
                
            narrativity_metrics = {
                'syllables_per_word': -1, 'nouns_density': -1, 'verbs_density': 1,
                'adjectives_density': -1, 'adverbs_density': 1, 'pronouns_density': 1,
                'first_person_pronouns_density': 1, 'third_person_pronouns_density': 1,
                'word_frequency_log': 1, 'content_word_frequency_log': -1,
                'min_word_frequency_per_sentence': -1, 'average_age_of_acquisition': -1,
                'average_familiarity': 1, 'negations_density': 1,
                'modifiers_per_noun_phrase': -1, 'passive_constructions_density': -1,
                'pos_dissimilarity_between_sentences': 1 
            }
            
            metric_values = {}
            for metric_name in narrativity_metrics.keys():
                metric_value = getattr(self, metric_name)(text)
                if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                    return -1
                metric_values[metric_name] = metric_value
            
            # Calculate a raw composite score directly without z-scoring
            for metric_name, loading in narrativity_metrics.items():
                if loading == -1:
                    metric_values[metric_name] = -metric_values[metric_name]
            
            # Calculate the mean of the aligned values
            raw_score = sum(metric_values.values()) / len(metric_values)
            return raw_score
            
        except Exception as e:
            print(f"Error in narrativity_cox: {e}")
            return -1

    def word_concreteness_cox(self, text):
        
        if not text:
            return -1
        try:
            if len(text.split()) < 5:
                return -1
                
            concreteness_metrics = {
                'average_concreteness': 1,
                'average_imagery': 1,
                'average_meaningfulness': 1
            }
            
            metric_values = {}
            for metric_name in concreteness_metrics.keys():
                metric_value = getattr(self, metric_name)(text)
                if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                    return -1
                metric_values[metric_name] = metric_value
            
            raw_score = sum(metric_values.values()) / len(metric_values)
            return raw_score
            
        except Exception as e:
            print(f"Error in word_concreteness_cox: {e}")
            return -1

    def referential_cohesion_cox(self, text):
       
        if not text:
            return -1
        try:
            if len(text.split()) < 10:
                return -1
                
            cohesion_metrics = {
                'content_word_overlap_adjacent': 1,
                'content_word_overlap_all': 1,
                'overlap_between_adjacent_sents_based_on_argument_bearing_words': 1,
                'argument_overlap_all': 1,
                'noun_overlap_adjacent': 1,
                'stem_overlap_all': 1,
                'type_token_ratio': -1,
                'mattr': -1,
                'verb_ttr': -1,
                'dissimilarity_of_words_between_sentences': -1
            }
            
            metric_values = {}
            for metric_name in cohesion_metrics.keys():
                metric_value = getattr(self, metric_name)(text)
                if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                    return -1
                metric_values[metric_name] = metric_value
            
            for metric_name, loading in cohesion_metrics.items():
                if loading == -1:
                    metric_values[metric_name] = -metric_values[metric_name]
            
            raw_score = sum(metric_values.values()) / len(metric_values)
            return raw_score
            
        except Exception as e:
            print(f"Error in referential_cohesion_cox: {e}")
            return -1
        
    def deep_causal_cohesion_cox(self, text):
        
        if not text:
            return -1
        try:
            if len(text.split()) < 10:
                return -1
                
            metric_groups = {
                'group1': ['number_of_connectives', 'causal_connectives', 'temporal_connectives', 'logical_connectives'],
                'group2': ['average_number_of_meaning_per_word', 'verb_overlap_adjacent'],
                'group3': ['additive_connectives', 'adversative_connectives'],
                'group4': ['temporal_cohesions', 'verb_tense_repetition', 'verb_aspect_repetition']
            }
            
            group_scores = {}
            for group_name, metrics in metric_groups.items():
                metric_values = {}
                for metric_name in metrics:
                    metric_value = getattr(self, metric_name)(text)
                    if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                        return -1
                    metric_values[metric_name] = metric_value
                
                if metric_values:
                    group_scores[group_name] = sum(metric_values.values()) / len(metric_values)
                else:
                    group_scores[group_name] = 0
            
            if group_scores:
                raw_score = sum(group_scores.values()) / len(group_scores)
                return raw_score
            else:
                return 0
            
        except Exception as e:
            print(f"Error in deep_causal_cohesion_cox: {e}")
            return -1
    
    def concept_density_concepts_per_sentence(self, text):
        if not text:
            return -1
        
        if not self.nlp:
            print("WARNING: spaCy not initialized in concept_density_concepts_per_sentence")
            return -1
        
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return -1

            total_concepts_per_sentence = 0
            sentence_count = 0
            
            for sentence in sentences:
                if not sentence.strip():  
                    continue
                
                concepts = self._get_concepts_from_text_conceptnet(sentence)
                distinct_concepts = len(set(concepts))
                total_concepts_per_sentence += distinct_concepts
                sentence_count += 1
            
            if sentence_count > 0:
                return total_concepts_per_sentence / sentence_count
            else:
                return 0
                
        except Exception as e:
            print(f"Error in concept_density_concepts_per_sentence: {e}")
            return -1

    def concept_density_concepts_per_sentence_dbpedia(self, text):
        if not text:
            return -1
        
        sentences = sent_tokenize(text)
        if not sentences:
            return -1

        entities = self._get_concepts_from_text_dbpedia(text)
        distinct_entities = len(set(entities))
        
        return distinct_entities / len(sentences) if sentences else 0

    def knowledge_graph_node_degree(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G or G.number_of_nodes() == 0:
            return -1
        
        if G.number_of_edges() == 0:
            return 0 
        
        degrees = [d for n, d in G.degree()]
        return sum(degrees) / G.number_of_nodes()

    def knowledge_graph_node_degree_dbpedia(self, text):
        G = self._build_text_graph_dbpedia(text)
        if not G or G.number_of_nodes() == 0:
            return -1
        
        if G.number_of_edges() == 0:
            return 0
        
        degrees = [d for n, d in G.degree()]
        return sum(degrees) / G.number_of_nodes()

    def knowledge_graph_node_clustering_coefficient(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G or G.number_of_nodes() < 3:
            return -1
        
        if G.number_of_edges() == 0:
            return 0
            
        try:
            return nx.average_clustering(G)
        except:
            return 0

    def knowledge_graph_node_clustering_coefficient_dbpedia(self, text):
        G = self._build_text_graph_dbpedia(text)
        if not G or G.number_of_nodes() < 3:
            return -1
        
        if G.number_of_edges() == 0:
            return 0
            
        try:
            return nx.average_clustering(G)
        except:
            return 0

    def knowledge_graph_average_node_pagerank(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G or G.number_of_nodes() == 0:
            return -1
        
        if G.number_of_edges() == 0:
            return 1.0 / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        try:
            pagerank = nx.pagerank(G, weight='weight')
            return sum(pagerank.values()) / len(pagerank) if pagerank else 0
        except:
            return 1.0 / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    def knowledge_graph_average_node_pagerank_dbpedia(self, text):
        G = self._build_text_graph_dbpedia(text)
        if not G or G.number_of_nodes() == 0:
            return -1
        
        if G.number_of_edges() == 0:
            return 1.0 / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        try:
            pagerank = nx.pagerank(G, weight='weight')  
            return sum(pagerank.values()) / len(pagerank) if pagerank else 0
        except:
            return 1.0 / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    def knowledge_graph_length_of_the_shortest_path(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G or G.number_of_nodes() < 2:
            return -1
        
        if G.number_of_edges() == 0:
            return 0  
        
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            
            if subgraph.number_of_nodes() < 2:
                return 0 

            return nx.average_shortest_path_length(subgraph, weight='weight')
        except:
            return -1

    def knowledge_graph_length_of_the_shortest_path_dbpedia(self, text):
        G = self._build_text_graph_dbpedia(text)
        if not G or G.number_of_nodes() < 2:
            return -1
        
        if G.number_of_edges() == 0:
            return 0  
        
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            
            if subgraph.number_of_nodes() < 2:
                return 0  

            return nx.average_shortest_path_length(subgraph, weight='weight') 
        except:
            return -1

    def knowledge_graph_exclusivity_based_semantic_relatedness(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G or G.number_of_nodes() < 2:
            return -1
            
        if G.number_of_edges() == 0:
            return 0  
            
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            
            if subgraph.number_of_nodes() < 2:
                return 0  
                
            exclusivity_scores = []
            
            for node1, node2, data in subgraph.edges(data=True):
                direct_relatedness = data.get('weight', 0)
                
                neighbors1 = set(subgraph.neighbors(node1)) - {node2}
                neighbors2 = set(subgraph.neighbors(node2)) - {node1}
                
                shared_neighbors = neighbors1.intersection(neighbors2)
                total_neighbors = len(neighbors1) + len(neighbors2)
                
                # Exclusivity: direct relatedness divided by proportion of shared connections
                # Higher values mean more exclusive relationship
                if total_neighbors > 0:
                    shared_ratio = len(shared_neighbors) / total_neighbors if total_neighbors > 0 else 0
                    exclusivity = direct_relatedness / (shared_ratio + 0.01)
                    exclusivity_scores.append(exclusivity)
            
            return sum(exclusivity_scores) / len(exclusivity_scores) if exclusivity_scores else 0
        except:
            return 0  

   

    def knowledge_graph_number_of_connected_components(self, text):
        G = self._build_text_graph_conceptnet(text)
        if not G:
            return -1
        return nx.number_connected_components(G)

    def knowledge_graph_number_of_connected_components_dbpedia(self, text):
        G = self._build_text_graph_dbpedia(text)
        if not G:
            return -1
        return nx.number_connected_components(G)
        
    def knowledge_graph_average_local_clustering_coefficient(self, text):
        return self.knowledge_graph_node_clustering_coefficient(text)

    def knowledge_graph_average_local_clustering_coefficient_dbpedia(self, text):
        return self.knowledge_graph_node_clustering_coefficient_dbpedia(text)

    def conceptual_graph_ontology_number_of_concpets(self, text):
        return len(self._get_concepts_from_text_conceptnet(text))

    def conceptual_graph_ontology_number_of_concpets_dbpedia(self, text):
        return len(self._get_concepts_from_text_dbpedia(text))
        
    def conceptual_graph_ontology_number_of_distinct_concepts(self, text):
        return len(set(self._get_concepts_from_text_conceptnet(text)))
    
    def conceptual_graph_ontology_number_of_distinct_concepts_dbpedia(self, text):
        return len(set(self._get_concepts_from_text_dbpedia(text)))
    
    def average_concreteness(self, text):
        if not text:
            return -1
        try:
            result = self.score_concreteness(text)
            if isinstance(result, (int, float)) and not isinstance(result, bool):
                return result
            else:
                return -1
        except Exception:
            return -1

    def ratio_of_concrete_to_abstract_words(self, text, threshold=3.0):
        if not text:
            return -1
        tokens = word_tokenize(text)
        if not tokens:
            return -1
        
        concrete_count = 0
        abstract_count = 0
        valid_words = 0
        
        for token in tokens:
            try:
                from sentence_concreteness.sentence_concreteness import get_concreteness
                score = get_concreteness(token.lower())
                if score is not None:  
                    valid_words += 1
                    if score > threshold:
                        concrete_count += 1
                    else:
                        abstract_count += 1
            except Exception:
                continue
        
        if valid_words == 0:
            return -1
        
        if abstract_count == 0:
            return 100 if concrete_count > 0 else 0
        
        if concrete_count == 0:
            return 0
        
        return concrete_count / abstract_count

    def _get_pos_word_lists(self, text, scope='sentence'):
        if not self.nlp or not text:
            return {}

        if scope == 'sentence':
            segments = self.nlp(text).sents
        elif scope == 'paragraph':
            if '\n\n' in text:
                paragraph_texts = [p.strip() for p in text.split('\n\n') if p.strip()]
            elif '\n' in text:
                paragraph_texts = [p.strip() for p in text.split('\n') if p.strip()]
            elif '\\p' in text:
                paragraph_texts = [p.strip() for p in text.split('\\p') if p.strip()]
            else:
                paragraph_texts = [text.strip()] if text.strip() else []

            segments = [self.nlp(p) for p in paragraph_texts if p]
        else:
            return {}
            
        lists = {
            'lemma': [], 'content': [], 'function': [], 'noun': [], 'verb': [],
            'adj': [], 'adv': [], 'pronoun': [], 'argument': []
        }
        
        for segment in segments:
            if not segment: continue

            lemma_list, content_list, function_list, noun_list, verb_list, adj_list, adv_list, pronoun_list, argument_list = [], [], [], [], [], [], [], [], []

            for token in segment:
                if token.is_punct:
                    continue
                
                lemma = token.lemma_.lower()
                
                lemma_list.append(lemma)
                
                if token.pos_ in ['NOUN', 'PROPN']:
                    noun_list.append(lemma)
                if token.pos_ == 'VERB':
                    verb_list.append(lemma)
                if token.pos_ == 'ADJ':
                    adj_list.append(lemma)
                if token.pos_ == 'ADV':
                    adv_list.append(lemma)
                if token.pos_ == 'PRON':
                    pronoun_list.append(lemma)
                if not token.is_stop and token.is_alpha:
                    content_list.append(lemma)
                if token.is_stop and token.is_alpha:
                    function_list.append(lemma)

            argument_list = noun_list + pronoun_list
            
            lists['lemma'].append(lemma_list)
            lists['content'].append(content_list)
            lists['function'].append(function_list)
            lists['noun'].append(noun_list)
            lists['verb'].append(verb_list)
            lists['adj'].append(adj_list)
            lists['adv'].append(adv_list)
            lists['pronoun'].append(pronoun_list)
            lists['argument'].append(argument_list)

        return lists

    def _calculate_adjacent_overlap(self, text, word_type, scope='sentence'):
        """Helper to calculate adjacent overlap for different word types and scopes."""
        word_lists = self._get_pos_word_lists(text, scope=scope).get(word_type, [])
        
        if len(word_lists) < 2:
            return 0
            
        total_overlap = 0
        total_unique_words_in_first_segments = 0
        
        for i in range(len(word_lists) - 1):
            set1 = set(word_lists[i])
            set2 = set(word_lists[i+1])
            
            if not set1:
                continue
                
            total_overlap += len(set1.intersection(set2))
            total_unique_words_in_first_segments += len(set1)
            
        return total_overlap / total_unique_words_in_first_segments if total_unique_words_in_first_segments > 0 else 0

    def overlap_between_adjacent_sents_based_on_argument_bearing_words(self, text):
        """Measures overlap of argument-bearing words (nouns and pronouns) between adjacent sentences."""
        return self._calculate_adjacent_overlap(text, 'argument', 'sentence')

    def verb_aspect_repetition(self, text):
        """Calculate verb aspect repetition by tracking aspect overlap between adjacent sentences."""
        if not self.nlp or not text:
            return -1
        
        doc = self.nlp(text)
        sents = list(doc.sents)
        if len(sents) < 2:
            return -1
            
        aspects_by_sentence = []
        for sent in sents:
            aspects = set()
            for token in sent:
                if token.pos_ == 'VERB':
                    aspect = token.morph.get('Aspect')
                    if aspect:
                        aspects.update(aspect)
            aspects_by_sentence.append(aspects)
        
        overlap_scores = []
        for i in range(len(aspects_by_sentence) - 1):
            aspects1 = aspects_by_sentence[i]
            aspects2 = aspects_by_sentence[i+1]
            
            if not aspects1 or not aspects2:
                continue

            intersection = len(aspects1.intersection(aspects2))
            union = len(aspects1.union(aspects2))
            
            if union > 0:
                overlap_scores.append(intersection / union)
        
        return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

    def temporal_cohesions(self, text):
        return self.temporal_connectives(text)

    def pos_dissimilarity_between_sentences(self, text):
        """
        Calculate the dissimilarity of parts of speech between adjacent sentences.
        Dissimilarity is defined as the Euclidean distance between vectors of POS tag frequencies.
        """
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        sents = list(doc.sents)
        if len(sents) < 2:
            return 0
        
        pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
                   'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        
        sent_pos_vectors = []
        for sent in sents:
            pos_counts = {pos: 0 for pos in pos_tags}
            total_tokens = len(sent)
            if total_tokens == 0:
                continue
                
            for token in sent:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
                    
            pos_freqs = {pos: count / total_tokens for pos, count in pos_counts.items()}
            sent_pos_vectors.append(pos_freqs)
        
        if len(sent_pos_vectors) < 2:
            return 0
            
        distances = []
        for i in range(len(sent_pos_vectors) - 1):
            vec1 = sent_pos_vectors[i]
            vec2 = sent_pos_vectors[i + 1]
            
            squared_diff_sum = sum((vec1.get(pos, 0) - vec2.get(pos, 0)) ** 2 for pos in pos_tags)
            distance = np.sqrt(squared_diff_sum)
            distances.append(distance)
            
        return np.mean(distances) if distances else 0


    def content_word_overlap_adjacent(self, text):
        """Content word overlap between adjacent sentences."""
        return self._calculate_adjacent_overlap(text, 'content', 'sentence')

    def content_word_overlap_all(self, text):
        """Content word overlap across all sentences in paragraph."""
        return self._calculate_adjacent_overlap(text, 'content', 'paragraph')

    def argument_overlap_adjacent(self, text):
        """Argument overlap between adjacent sentences."""
        return self._calculate_adjacent_overlap(text, 'argument', 'sentence')

    def argument_overlap_all(self, text):
        """Argument overlap across all sentences in paragraph."""
        return self._calculate_adjacent_overlap(text, 'argument', 'paragraph')

    def noun_overlap_adjacent(self, text):
        """Noun overlap between adjacent sentences."""
        return self._calculate_adjacent_overlap(text, 'noun', 'sentence')

    def stem_overlap_all(self, text):
        """Stem overlap between paragraphs using Porter stemmer."""
        if not text:
            return 0
        
        try:
            stemmer = PorterStemmer()
        except ImportError:
            return 0
        
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 2:
            return 0
        
        stemmed_paragraphs = []
        for paragraph in paragraphs:
            if paragraph.strip():
                sentences = sent_tokenize(paragraph.strip())
                all_stems = set()
                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    stems = [stemmer.stem(word) for word in words if word.isalpha()]
                    all_stems.update(stems)
                stemmed_paragraphs.append(all_stems)
        
        if len(stemmed_paragraphs) < 2:
            return 0
        
        total_overlap = 0
        total_pairs = 0
        
        for i in range(len(stemmed_paragraphs)):
            for j in range(i + 1, len(stemmed_paragraphs)):
                stems1 = stemmed_paragraphs[i]
                stems2 = stemmed_paragraphs[j]
                
                if stems1 and stems2:
                    intersection = len(stems1.intersection(stems2))
                    union = len(stems1.union(stems2))
                    if union > 0:
                        overlap_ratio = intersection / union
                        total_overlap += overlap_ratio
                    total_pairs += 1
        
        return total_overlap / total_pairs if total_pairs > 0 else 0

    def stem_overlap_sent(self, text):
        """Average stem overlap between adjacent sentences using Porter stemmer."""
        if not text:
            return 0
        
        try:
            stemmer = PorterStemmer()
        except ImportError:
            return 0
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0
        
        stemmed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            stems = [stemmer.stem(word) for word in words if word.isalpha()]
            stemmed_sentences.append(set(stems))
        
        total_overlap = 0
        total_pairs = 0
        
        for i in range(len(stemmed_sentences) - 1):
            stems1 = stemmed_sentences[i]
            stems2 = stemmed_sentences[i + 1]
            
            if stems1 and stems2:
                intersection = len(stems1.intersection(stems2))
                union = len(stems1.union(stems2))
                if union > 0:
                    overlap_ratio = intersection / union
                    total_overlap += overlap_ratio
                total_pairs += 1
        
        return total_overlap / total_pairs if total_pairs > 0 else 0

    def function_word_mattr(self, text):
        """Moving-Average Type-Token Ratio for function words."""
        if not self.nlp or not text:
            return -1
        doc = self.nlp(text)
        function_words = [token.lemma_.lower() for token in doc if token.is_stop and token.is_alpha]
        if len(function_words) < 50: # MATTR default window is 50
            return len(set(function_words)) / len(function_words) if function_words else 0
        return taaled.lexdiv(function_words).mattr

    def syllables_per_word(self, text):
        """Average number of syllables per word."""
        if not text:
            return 0
        words = word_tokenize(text)
        if not words:
            return 0
        
        total_syllables = 0
        word_count = 0
        for word in words:
            if word.isalpha():
                try:
                    total_syllables += syllable_count(word)
                except (KeyError, Exception):
                    vowels = "aeiouy"
                    word = word.lower()
                    count = 0
                    prev_is_vowel = False
                    for char in word:
                        is_vowel = char in vowels
                        if is_vowel and not prev_is_vowel:
                            count += 1
                        prev_is_vowel = is_vowel
                    if word.endswith('e'):
                        count -= 1
                    if count == 0:
                        count = 1
                    total_syllables += count
                word_count += 1
                
        return total_syllables / word_count if word_count > 0 else 0

    def nouns_density(self, text):
        """Proportion of nouns in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        nouns = len([token for token in doc if token.pos_ == 'NOUN' and token.is_alpha])
        return nouns / total_words if total_words > 0 else 0

    def verbs_density(self, text):
        """Proportion of verbs in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        verbs = len([token for token in doc if token.pos_ == 'VERB' and token.is_alpha])
        return verbs / total_words if total_words > 0 else 0

    def adjectives_density(self, text):
        """Proportion of adjectives in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        adjectives = len([token for token in doc if token.pos_ == 'ADJ' and token.is_alpha])
        return adjectives / total_words if total_words > 0 else 0

    def adverbs_density(self, text):
        """Proportion of adverbs in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        adverbs = len([token for token in doc if token.pos_ == 'ADV' and token.is_alpha])
        return adverbs / total_words if total_words > 0 else 0

    def pronouns_density(self, text):
        """Proportion of pronouns in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        pronouns = len([token for token in doc if token.pos_ == 'PRON' and token.is_alpha])
        return pronouns / total_words if total_words > 0 else 0

    def first_person_pronouns_density(self, text):
        """Proportion of first person pronouns in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        first_person_count = len([token for token in doc if token.text.lower() in first_person and token.is_alpha])
        return first_person_count / total_words if total_words > 0 else 0

    def third_person_pronouns_density(self, text):
        """Proportion of third person pronouns in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        third_person = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']
        third_person_count = len([token for token in doc if token.text.lower() in third_person and token.is_alpha])
        return third_person_count / total_words if total_words > 0 else 0

    def word_frequency_log(self, text):
        """Average logarithm of word frequency."""
        if not text:
            return 0
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        if not words:
            return 0
        
        frequencies = []
        for word in words:
            freq = self.word_freq_db.get(word, 1)  
            if freq > 0:
                frequencies.append(np.log(freq))
            else:
                frequencies.append(np.log(1))  
        
        return np.mean(frequencies) if frequencies else 0

    def content_word_frequency_log(self, text):
        """Average logarithm of content word frequency."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        content_words = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        
        if not content_words:
            return 0
        
        frequencies = []
        for word in content_words:
            freq = self.word_freq_db.get(word, 1)
            if freq > 0:
                frequencies.append(np.log(freq))
            else:
                frequencies.append(np.log(1))
        
        return np.mean(frequencies) if frequencies else 0

    def min_word_frequency_per_sentence(self, text):
        """Minimum word frequency per sentence (averaged across sentences)."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0
        
        min_frequencies = []
        for sent in sentences:
            words = [token.text.lower() for token in sent if token.is_alpha]
            if words:
                frequencies = [self.word_freq_db.get(word, 1) for word in words]
                min_frequencies.append(min(frequencies))
        
        return np.mean(min_frequencies) if min_frequencies else 0

    def average_age_of_acquisition(self, text):
        """Average age of acquisition scores for words in text."""
        scores = self._get_psycholinguistic_scores(text, self.kuperman_aoa)
        return np.mean(scores) if scores else 0

    def average_familiarity(self, text):
        """Average familiarity scores for words in text."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_familiarity)
        return np.mean(scores) if scores else 0

    def average_imagery(self, text):
        """Average imagery scores for words in text."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_imagery)
        return np.mean(scores) if scores else 0

    def average_meaningfulness(self, text):
        """Average meaningfulness scores for words in text."""
        scores = self._get_psycholinguistic_scores(text, self.mrc_meaningfulness)
        return np.mean(scores) if scores else 0

    def negations_density(self, text):
        """Proportion of negation words in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_words = len([token for token in doc if token.is_alpha])
        negations = ['no', 'not', 'none', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'cannot', "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"]
        negation_count = len([token for token in doc if token.text.lower() in negations and token.is_alpha])
        return negation_count / total_words if total_words > 0 else 0

    def modifiers_per_noun_phrase(self, text):
        """Average number of modifiers per noun phrase."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        
        noun_phrases = 0
        total_modifiers = 0
        
        for token in doc:
            if token.pos_ == 'NOUN':
                noun_phrases += 1
                modifiers = [child for child in token.children if child.pos_ in ['ADJ', 'DET']]
                total_modifiers += len(modifiers)
        
        return total_modifiers / noun_phrases if noun_phrases > 0 else 0

    def passive_constructions_density(self, text):
        """Proportion of passive constructions in the text."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        total_verbs = len([token for token in doc if token.pos_ == 'VERB'])
        
        passive_count = 0
        for token in doc:
            if token.tag_ == 'VBN':  # Past participle
                # Check if there's an auxiliary verb nearby
                for child in token.children:
                    if child.lemma_ in ['be', 'get'] and child.pos_ == 'AUX':
                        passive_count += 1
                        break
        
        return passive_count / total_verbs if total_verbs > 0 else 0

    def mattr(self, text):
        """Moving Average Type-Token Ratio using TAACO."""
        if not text:
            return 0
        tokens = word_tokenize(text)
        if len(tokens) < 50:
            return len(set(tokens)) / len(tokens) if tokens else 0
        return taaled.lexdiv(tokens).mattr

    def verb_ttr(self, text):
        """Type-Token Ratio for verbs using TAACO."""
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        verbs = [token.lemma_.lower() for token in doc if token.pos_ == 'VERB']
        if not verbs:
            return 0
        return len(set(verbs)) / len(verbs)

    def _count_specific_connectives(self, text, connective_type):
        """Count specific types of connectives."""
        if not text or connective_type not in CONNECTIVES:
            return 0
        
        connectives = CONNECTIVES[connective_type]
        text_lower = text.lower()
        
        count = 0
        for connective in connectives:
            pattern = r'\b' + re.escape(connective.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        
        return count

    def connectives(self, text):
        """Total number of connectives using TAACO (alias for number_of_connectives)."""
        return self.number_of_connectives(text)

    def causal_connectives(self, text):
        """Count causal connectives in the text using CONNECTIVES['all_causal']."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['all_causal']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def temporal_connectives(self, text):
        """Count temporal connectives in the text using CONNECTIVES['all_temporal']."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['all_temporal']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def logical_connectives(self, text):
        """Count logical connectives in the text using CONNECTIVES['all_logical']."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['all_logical']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def additive_connectives(self, text):
        """Count additive connectives in the text using CONNECTIVES['all_additive']."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['all_additive']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def adversative_connectives(self, text):
        """Count adversative connectives in the text using CONNECTIVES['opposition']."""
        if not text:
            return 0
        from connectives_list import CONNECTIVES
        connectives = CONNECTIVES['opposition']
        text_lower = text.lower()
        count = 0
        for conn in connectives:
            pattern = r'\b' + re.escape(conn.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        return count

    def verb_overlap_adjacent(self, text):
        """Verb overlap between adjacent sentences."""
        return self._calculate_adjacent_overlap(text, 'verb', 'sentence')

    def verb_tense_repetition(self, text):
        """Measure verb tense repetition between adjacent sentences using spaCy."""
        if not self.nlp or not text:
            return 0
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 0
        
        tense_matches = 0
        total_pairs = 0
        
        for i in range(len(sentences) - 1):
            sent1_tenses = [token.tag_ for token in sentences[i] if token.pos_ == 'VERB']
            sent2_tenses = [token.tag_ for token in sentences[i + 1] if token.pos_ == 'VERB']
            
            if sent1_tenses and sent2_tenses:
                if any(tense in sent2_tenses for tense in sent1_tenses):
                    tense_matches += 1
                total_pairs += 1
        
        return tense_matches / total_pairs if total_pairs > 0 else 0

    def verb_tense_repetition_nltk(self, text):
        """Measure verb tense repetition between adjacent sentences using NLTK averaged_perceptron_tagger_eng."""
        if not text:
            return 0
        
        try:
            import nltk
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 0
            
            tense_tags_by_sentence = []
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tags = nltk.pos_tag(tokens)
                tenses = set()
                for word, tag in tags:
                    if tag.startswith('VB'):
                        tenses.add(tag)
                tense_tags_by_sentence.append(tenses)
            
            tense_matches = 0
            total_pairs = 0
            for i in range(len(tense_tags_by_sentence) - 1):
                tenses1 = tense_tags_by_sentence[i]
                tenses2 = tense_tags_by_sentence[i + 1]
                if tenses1 and tenses2:
                    if any(tense in tenses2 for tense in tenses1):
                        tense_matches += 1
                    total_pairs += 1
            
            return tense_matches / total_pairs if total_pairs > 0 else 0
        except ImportError:
            print("WARNING: NLTK not available for verb_tense_repetition_nltk")
            return -1
        except LookupError as e:
            print(f"WARNING: NLTK tagger not found for verb_tense_repetition_nltk: {e}")
            return -1

    def dissimilarity_of_words_between_sentences(self, text):
        """
        Calculate the dissimilarity of words between adjacent sentences.
        This measures how different the vocabulary is between adjacent sentences.
        """
        if not self.nlp or not text:
            return 0
        doc = self.nlp(text)
        sents = list(doc.sents)
        if len(sents) < 2:
            return 0

        dissimilarity_scores = []
        for i in range(len(sents) - 1):
            words1 = {token.lemma_.lower() for token in sents[i] if token.is_alpha and not token.is_stop}
            words2 = {token.lemma_.lower() for token in sents[i+1] if token.is_alpha and not token.is_stop}
            
            if not words1 or not words2:
                continue
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            if union > 0:
                jaccard_similarity = intersection / union
                dissimilarity = 1 - jaccard_similarity
                dissimilarity_scores.append(dissimilarity)
        
        return np.mean(dissimilarity_scores) if dissimilarity_scores else 0

 

    def polysemy(self, text):
        return self.average_number_of_meaning_per_word(text)

    def calculate_all_metrics(self, text):
        
        results = {}
        for _, row in self.metrics_df.iterrows():
            metric_name = row['Metric']
            function_name = self._clean_metric_name(metric_name)
            if hasattr(self, function_name):
                try:
                    results[metric_name] = getattr(self, function_name)(text)
                except Exception as e:
                    print(f"Error calculating metric '{metric_name}': {e}")
                    results[metric_name] = "Error"
            else:
                results[metric_name] = "Not Implemented"
        return results

    def calculate_narrativity_score(self, texts):
        """
        Calculates the narrativity score for a list of texts.
        If only one text is provided, returns the raw composite score (not z-scored).
        """
        narrativity_metrics = {
            'syllables_per_word': -1, 'nouns_density': -1, 'verbs_density': 1,
            'adjectives_density': -1, 'adverbs_density': 1, 'pronouns_density': 1,
            'first_person_pronouns_density': 1, 'third_person_pronouns_density': 1,
            'word_frequency_log': 1, 'content_word_frequency_log': -1,
            'min_word_frequency_per_sentence': -1, 'average_age_of_acquisition': -1,
            'average_familiarity': 1, 'negations_density': 1,
            'modifiers_per_noun_phrase': -1, 'passive_constructions_density': -1,
            'pos_dissimilarity_between_sentences': 1  
        }
        all_metrics_data = []
        for text in texts:
            metrics = {name: getattr(self, name)(text) for name in narrativity_metrics}
            all_metrics_data.append(metrics)
        metrics_df = pd.DataFrame(all_metrics_data)
        # Standardize (z-scores)
        for col in metrics_df.columns:
            mean = metrics_df[col].mean()
            std = metrics_df[col].std()
            if std > 0:
                metrics_df[col] = (metrics_df[col] - mean) / std
            else:
                metrics_df[col] = 0
        for metric, loading in narrativity_metrics.items():
            if loading == -1:
                metrics_df[metric] *= -1
        # Compute component score (unweighted mean of aligned z-scores)
        narrativity_scores = metrics_df.mean(axis=1)
        # Re-standardize (only if more than one text)
        if len(texts) > 1:
            mean_score = narrativity_scores.mean()
            std_score = narrativity_scores.std()
            if std_score > 0:
                return (narrativity_scores - mean_score) / std_score
            else:
                return pd.Series([0] * len(texts))
        else:
            return narrativity_scores

    def calculate_referential_cohesion_score(self, texts):
        """
        Calculates the referential cohesion score for a list of texts.
        If only one text is provided, returns the raw composite score (not z-scored).
        """
        cohesion_metrics = {
            'content_word_overlap_adjacent': 1,
            'content_word_overlap_all': 1,
            'overlap_between_adjacent_sents_based_on_argument_bearing_words': 1,
            'argument_overlap_all': 1,
            'noun_overlap_adjacent': 1,
            'stem_overlap_all': 1,
            'type_token_ratio': -1,
            'mattr': -1,  
            'verb_ttr': -1,  
            'dissimilarity_of_words_between_sentences': -1  
        }
        all_metrics_data = []
        for text in texts:
            metrics = {name: getattr(self, name)(text) for name in cohesion_metrics}
            all_metrics_data.append(metrics)
        metrics_df = pd.DataFrame(all_metrics_data)
        for col in metrics_df.columns:
            mean = metrics_df[col].mean()
            std = metrics_df[col].std()
            if std > 0:
                metrics_df[col] = (metrics_df[col] - mean) / std
            else:
                metrics_df[col] = 0
        for metric, loading in cohesion_metrics.items():
            if loading == -1:
                metrics_df[metric] *= -1
        cohesion_scores = metrics_df.mean(axis=1)
        if len(texts) > 1:
            mean_score = cohesion_scores.mean()
            std_score = cohesion_scores.std()
            if std_score > 0:
                return (cohesion_scores - mean_score) / std_score
            else:
                return pd.Series([0] * len(texts))
        else:
            return cohesion_scores

    def calculate_word_concreteness_score(self, texts):
        """
        Calculates the word concreteness score for a list of texts.
        If only one text is provided, returns the raw composite score (not z-scored).
        """
        concreteness_metrics = {
            'average_concreteness': 1,
            'average_imagery': 1,
            'average_meaningfulness': 1
        }
        all_metrics_data = []
        for text in texts:
            metrics = {name: getattr(self, name)(text) for name in concreteness_metrics}
            all_metrics_data.append(metrics)
        metrics_df = pd.DataFrame(all_metrics_data)
        for col in metrics_df.columns:
            mean = metrics_df[col].mean()
            std = metrics_df[col].std()
            if std > 0:
                metrics_df[col] = (metrics_df[col] - mean) / std
            else:
                metrics_df[col] = 0
        concreteness_scores = metrics_df.mean(axis=1)
        if len(texts) > 1:
            mean_score = concreteness_scores.mean()
            std_score = concreteness_scores.std()
            if std_score > 0:
                return (concreteness_scores - mean_score) / std_score
            else:
                return pd.Series([0] * len(texts))
        else:
            return concreteness_scores

    def calculate_deep_causal_cohesion_score(self, texts):
        """
        Calculates the deep causal cohesion score by averaging the scores of four component groups.
        If only one text is provided, returns the raw composite score (not z-scored).
        """
        metric_groups = {
            'group1': ['number_of_connectives', 'causal_connectives', 'temporal_connectives', 'logical_connectives'],
            'group2': ['average_number_of_meaning_per_word', 'verb_overlap_adjacent'],
            'group3': ['additive_connectives', 'adversative_connectives'],
            'group4': ['temporal_cohesions', 'verb_tense_repetition', 'verb_aspect_repetition']
        }
        group_scores = pd.DataFrame()
        for group, metrics in metric_groups.items():
            all_metrics_data = []
            for text in texts:
                text_metrics = {name: getattr(self, name)(text) for name in metrics}
                all_metrics_data.append(text_metrics)
            metrics_df = pd.DataFrame(all_metrics_data)
            for col in metrics_df.columns:
                mean = metrics_df[col].mean()
                std = metrics_df[col].std()
                if std > 0:
                    metrics_df[col] = (metrics_df[col] - mean) / std
                else:
                    metrics_df[col] = 0
            group_scores[group] = metrics_df.mean(axis=1)
        final_scores = group_scores.mean(axis=1)
        if len(texts) > 1:
            mean_score = final_scores.mean()
            std_score = final_scores.std()
            if std_score > 0:
                return (final_scores - mean_score) / std_score
            else:
                return pd.Series([0] * len(texts))
        else:
            return final_scores

    def process_dataset(self, dataset, output_csv_path):
        """
        Processes a dataset of texts and saves the complexity metrics to a CSV file.
        Uses all implemented methods directly instead of reading from Metrics.csv.
        """
        all_results = []
        texts_for_narrativity = []

        excluded_methods = [
            'process_dataset', 'calculate_all_metrics', 'calculate_metrics_by_type',
            'calculate_narrativity_score', 'calculate_referential_cohesion_score', 
            'calculate_word_concreteness_score', 'calculate_deep_causal_cohesion_score'
        ]
        
        metric_methods = [method for method in dir(self) 
                        if callable(getattr(self, method)) 
                        and not method.startswith('_')
                        and method not in excluded_methods]

        total_texts = len(dataset)
        print(f"Starting to process {total_texts} texts...", flush=True)

        for i, item in enumerate(dataset):
            if (i + 1) % 500 == 0:
                print(f"Progress: Processed {i + 1}/{total_texts} texts ({(i + 1)/total_texts*100:.1f}%)", flush=True)
            
            text = item.get('text', '')
            if not text:
                print(f"WARNING: Skipping item with empty text (ID: {item.get('id', 'N/A')}).", flush=True)
                all_results.append({'id': item.get('id'), 'input': ''})
                texts_for_narrativity.append("")  
                continue

            texts_for_narrativity.append(text)
            
            row = {'id': item.get('id'), 'input': text}
            for method_name in metric_methods:
                try:
                    method = getattr(self, method_name)
                    result = method(text)
                    row[method_name] = result
                except Exception as e:
                    print(f"Error calculating {method_name}: {e}")
                    row[method_name] = -1
            
            all_results.append(row)

        if not all_results:
            print("No texts were processed. Output CSV will not be created.")
            return

        results_df = pd.DataFrame(all_results)
        
        if texts_for_narrativity:
            non_empty_texts = [text for text in texts_for_narrativity if text.strip()]
            non_empty_indices = [i for i, text in enumerate(texts_for_narrativity) if text.strip()]
            
            if non_empty_texts:
                try:
                    narrativity_scores = self.calculate_narrativity_score(non_empty_texts)
                    narrativity_column = [None] * len(texts_for_narrativity)
                    for idx, score in zip(non_empty_indices, narrativity_scores):
                        narrativity_column[idx] = score
                    results_df['narrativity_score'] = narrativity_column
                except Exception as e:
                    print(f"Error calculating narrativity scores: {e}")
                    results_df['narrativity_score'] = [None] * len(texts_for_narrativity)

                try:
                    referential_cohesion_scores = self.calculate_referential_cohesion_score(non_empty_texts)
                    referential_column = [None] * len(texts_for_narrativity)
                    for idx, score in zip(non_empty_indices, referential_cohesion_scores):
                        referential_column[idx] = score
                    results_df['referential_cohesion_score'] = referential_column
                except Exception as e:
                    print(f"Error calculating referential cohesion scores: {e}")
                    results_df['referential_cohesion_score'] = [None] * len(texts_for_narrativity)

                try:
                    word_concreteness_scores = self.calculate_word_concreteness_score(non_empty_texts)
                    concreteness_column = [None] * len(texts_for_narrativity)
                    for idx, score in zip(non_empty_indices, word_concreteness_scores):
                        concreteness_column[idx] = score
                    results_df['word_concreteness_score'] = concreteness_column
                except Exception as e:
                    print(f"Error calculating word concreteness scores: {e}")
                    results_df['word_concreteness_score'] = [None] * len(texts_for_narrativity)
                    
                try:
                    deep_causal_cohesion_scores = self.calculate_deep_causal_cohesion_score(non_empty_texts)
                    causal_column = [None] * len(texts_for_narrativity)
                    for idx, score in zip(non_empty_indices, deep_causal_cohesion_scores):
                        causal_column[idx] = score
                    results_df['deep_causal_cohesion_score'] = causal_column
                except Exception as e:
                    print(f"Error calculating deep causal cohesion scores: {e}")
                    results_df['deep_causal_cohesion_score'] = [None] * len(texts_for_narrativity)
            else:
                results_df['narrativity_score'] = [None] * len(texts_for_narrativity)
                results_df['referential_cohesion_score'] = [None] * len(texts_for_narrativity)
                results_df['word_concreteness_score'] = [None] * len(texts_for_narrativity)
                results_df['deep_causal_cohesion_score'] = [None] * len(texts_for_narrativity)

        cols = ['id', 'input'] + [col for col in results_df.columns if col not in ['id', 'input']]
        results_df = results_df[cols]

        try:
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Successfully saved results for {len(all_results)} texts to '{output_csv_path}'.")
        except Exception as e:
            print(f"Error saving CSV file to '{output_csv_path}': {e}")

    


    def calculate_metrics_by_type(self, text, metric_type):
        """
        Calculates all metrics of a specific type (e.g., 'Lexical', 'Structural').
        """
        results = {}
        
        filtered_df = self.metrics_df[self.metrics_df['Type'].str.lower() == metric_type.lower()]

        if filtered_df.empty:
            print(f"WARNING: No metrics found for type '{metric_type}'.")
            return results

        for _, row in filtered_df.iterrows():
            metric_name = row['Metric']
            function_name = self._clean_metric_name(metric_name)
            if hasattr(self, function_name):
                try:
                    results[metric_name] = getattr(self, function_name)(text)
                except Exception as e:
                    print(f"Error calculating metric '{metric_name}': {e}")
                    results[metric_name] = "Error"
            else:
                results[metric_name] = "Not Implemented"
        return results

    def coca_academic_frequency_score(self, text):
        """Average COCA academic frequency for lemmatized words in the text, using COCA_acad_word_list_lemma_freq.csv column 2."""
        if not text or not self.nlp:
            return -1
        doc = self.nlp(text)
        scores = [self.coca_academic_frequency.get(token.lemma_.lower(), 0) for token in doc if token.lemma_.lower() in self.coca_academic_frequency]
        return np.mean(scores) if scores else 0

    def referential_cohesion_cox(self, text):
        """Returns the referential cohesion score for a single text.
        Note: For accurate scores, use calculate_referential_cohesion_score() with multiple texts."""
        if not text:
            return -1
        try:
            if len(text.split()) < 10:
                return -1
                
            cohesion_metrics = {
                'content_word_overlap_adjacent': 1,
                'content_word_overlap_all': 1,
                'overlap_between_adjacent_sents_based_on_argument_bearing_words': 1,
                'argument_overlap_all': 1,
                'noun_overlap_adjacent': 1,
                'stem_overlap_all': 1,
                'type_token_ratio': -1,
                'mattr': -1,
                'verb_ttr': -1,
                'dissimilarity_of_words_between_sentences': -1
            }
            
            metric_values = {}
            for metric_name in cohesion_metrics.keys():
                metric_value = getattr(self, metric_name)(text)
                if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                    return -1
                metric_values[metric_name] = metric_value
            
            for metric_name, loading in cohesion_metrics.items():
                if loading == -1:
                    metric_values[metric_name] = -metric_values[metric_name]
            
            raw_score = sum(metric_values.values()) / len(metric_values)
            return raw_score
            
        except Exception as e:
            print(f"Error in referential_cohesion_cox: {e}")
            return -1

    def deep_causal_cohesion_cox(self, text):
        """Returns the deep causal cohesion score for a single text.
        Note: For accurate scores, use calculate_deep_causal_cohesion_score() with multiple texts."""
        if not text:
            return -1
        try:
            if len(text.split()) < 10:
                return -1
                
            metric_groups = {
                'group1': ['number_of_connectives', 'causal_connectives', 'temporal_connectives', 'logical_connectives'],
                'group2': ['average_number_of_meaning_per_word', 'verb_overlap_adjacent'],
                'group3': ['additive_connectives', 'adversative_connectives'],
                'group4': ['temporal_cohesions', 'verb_tense_repetition', 'verb_aspect_repetition']
            }
            
            group_scores = {}
            for group_name, metrics in metric_groups.items():
                metric_values = {}
                for metric_name in metrics:
                    metric_value = getattr(self, metric_name)(text)
                    if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                        return -1
                    metric_values[metric_name] = metric_value
                
                if metric_values:
                    group_scores[group_name] = sum(metric_values.values()) / len(metric_values)
                else:
                    group_scores[group_name] = 0
            
            if group_scores:
                raw_score = sum(group_scores.values()) / len(group_scores)
                return raw_score
            else:
                return 0
            
        except Exception as e:
            print(f"Error in deep_causal_cohesion_cox: {e}")
            return -1

    def deep_causal_cohesion_cox_nltk(self, text):
        """Returns the deep causal cohesion score for a single text using NLTK-based methods.
        Note: For accurate scores, use calculate_deep_causal_cohesion_score() with multiple texts."""
        if not text:
            return -1
        try:
            if len(text.split()) < 10:
                return -1
                
            metric_groups = {
                'group1': ['number_of_connectives', 'causal_connectives', 'temporal_connectives', 'logical_connectives'],
                'group2': ['average_number_of_meaning_per_word', 'verb_overlap_adjacent'],
                'group3': ['additive_connectives', 'adversative_connectives'],
                'group4': ['temporal_cohesions', 'verb_tense_repetition_nltk', 'verb_aspect_repetition']
            }
            
            group_scores = {}
            for group_name, metrics in metric_groups.items():
                metric_values = {}
                for metric_name in metrics:
                    metric_value = getattr(self, metric_name)(text)
                    if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
                        return -1
                    metric_values[metric_name] = metric_value
                
                if metric_values:
                    group_scores[group_name] = sum(metric_values.values()) / len(metric_values)
                else:
                    group_scores[group_name] = 0
            
            if group_scores:
                raw_score = sum(group_scores.values()) / len(group_scores)
                return raw_score
            else:
                return 0
            
        except Exception as e:
            print(f"Error in deep_causal_cohesion_cox_nltk: {e}")
            return -1

    def _debug_graph_building(self, text, method='conceptnet'):
        """[Debug method] Analyze graph building process to understand why graphs might be empty."""
        print(f"\n=== DEBUG: Graph Building Analysis ({method.upper()}) ===")
        print(f"Input text: {text[:100]}...")
        
        if method == 'conceptnet':
            # Analyze ConceptNet graph building
            if not self.nlp:
                print("ERROR: spaCy NLP model not loaded")
                return
            
            doc = self.nlp(text)
            concepts = [
                token.lemma_.lower() for token in doc
                if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop
            ]
            print(f"Extracted concepts: {concepts}")
            
            if not concepts:
                print("WARNING: No concepts extracted from text")
                return
            
            test_pairs = []
            for i in range(min(3, len(concepts))):
                for j in range(i+1, min(4, len(concepts))):
                    test_pairs.append((concepts[i], concepts[j]))
            
            print(f"Testing relatedness for pairs: {test_pairs}")
            for concept1, concept2 in test_pairs:
                relatedness = self._get_relatedness_conceptnet(concept1, concept2)
                print(f"  {concept1} <-> {concept2}: {relatedness}")
        
        elif method == 'dbpedia':
            entities = self._get_concepts_from_text_dbpedia(text)
            print(f"Extracted DBpedia entities: {entities}")
            
            if not entities:
                print("WARNING: No DBpedia entities extracted from text")
                return
            
            unique_entities = list(set(entities))
            print(f"Unique entities: {unique_entities}")
            
            if len(unique_entities) >= 2:
                uri1, uri2 = unique_entities[0], unique_entities[1]
                related = self.dbpedia.are_related(uri1, uri2)
                print(f"Testing relationship: {uri1} <-> {uri2}: {related}")
        
        # Build the actual graph
        if method == 'conceptnet':
            G = self._build_text_graph_conceptnet(text)
        else:
            G = self._build_text_graph_dbpedia(text)
        
        print(f"Graph statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Connected components: {nx.number_connected_components(G)}")
        
        if G.number_of_edges() > 0:
            print(f"  Sample edges: {list(G.edges(data=True))[:3]}")
        
        print("=== END DEBUG ===\n")

    def _build_text_graph_conceptnet(self, text, threshold=0.05):  
        """[Helper function] Build knowledge graph from text using ConceptNet."""
        text_hash = hash(text)
        if text_hash in self.graph_cache_conceptnet:
            return self.graph_cache_conceptnet[text_hash]

        if not self.nlp:
            return nx.Graph()

        doc = self.nlp(text)
        G = nx.Graph()
        
        concepts = set(
            token.lemma_.lower() for token in doc 
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop
        )
        G.add_nodes_from(concepts)

        checked_pairs = set()

        for token in doc:
            if token.is_stop or token.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                continue

            head = token.head
            if head.is_stop or head.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ'] or head == token:
                pass
            else:
                concept1 = token.lemma_.lower()
                concept2 = head.lemma_.lower()

                pair = tuple(sorted((concept1, concept2)))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    relatedness = self._get_relatedness_conceptnet(concept1, concept2)
                    if relatedness > threshold:  
                        G.add_edge(concept1, concept2, weight=relatedness)

            for sibling in token.head.children:
                if sibling == token or sibling.is_stop or sibling.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                    continue
            
                concept1 = token.lemma_.lower()
                concept2 = sibling.lemma_.lower()
                pair = tuple(sorted((concept1, concept2)))

                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    relatedness = self._get_relatedness_conceptnet(concept1, concept2)
                    if relatedness > threshold:  
                        G.add_edge(concept1, concept2, weight=relatedness)
        
        self.graph_cache_conceptnet[text_hash] = G
        return G

if __name__ == '__main__':
    sample_dataset = [
        {
            'id': 1,
            'text': (
                "Once upon a time, in a land far, far away, lived a princess in a grand castle. "
                "She had a kind heart and was loved by all her people. Every day, she would walk through the "
                "village, greeting everyone with a warm smile. I think she was a wonderful person."
            )
        },
        {
            'id': 2,
            'text': (
                "The scientific paper details a novel method for quantum computing. The process involves "
                "the manipulation of qubits in a controlled, isolated environment. This breakthrough could "
                "potentially revolutionize data encryption and computational simulations. We believe our findings "
                "are significant."
            )
        },
        {
            'id': 3,
            'text': (
                "Our trip to the mountains was an adventure. We hiked for miles and saw a bear. "
                "He was big and brown, and he didn't see us. We were very quiet. It was exciting, but also a little scary."
            )
        },
        {
            'id': 4,
            'text': (
                "Legislative procedures require that any new bill must first be introduced in either the House or "
                "the Senate. Following its introduction, it is assigned to a committee for review. The committee's "
                "analysis and subsequent report are crucial for the bill's progression."
            )
        },
        {
            'id': 'text_005',
            'text': "" 
        }
    ]

    ''' print("Initializing ComplexiMeter...")
    meter = ComplexiMeter(metrics_file='Metrics.csv', crat_path='CRAT_v1.1.app')
    print("ComplexiMeter Initialized.")
    print("-" * 30)

    print("Processing dataset and saving to CSV...")
    output_file = 'complexity_results.csv'
    meter.process_dataset(sample_dataset, output_file)
    print("-" * 30)
    
    try:
        results_df = pd.read_csv(output_file)
        print(f"Successfully created '{output_file}'. Here's a preview:")
        print(results_df.head().to_string())
    except FileNotFoundError:
        print(f"Could not find the output file '{output_file}' to display.")
    except Exception as e:
        print(f"An error occurred while reading the output file: {e}") '''
    
    print("Processing complexity data from parquet files...")
    # 
    from process_complexity_data import main as process_main
    # 
    process_main() 
    
    #print("\nProcessing conversation complexity data...")
    # 
    #from process_conversation_complexity import main as conversation_main
    # 
    #conversation_main()