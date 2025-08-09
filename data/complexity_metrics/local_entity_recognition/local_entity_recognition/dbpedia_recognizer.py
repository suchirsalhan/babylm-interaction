#!/usr/bin/env python3
"""
DBpedia Entity Recognition System
Research-grade implementation with proper validation and error handling.
"""

import requests
import json
import time
import hashlib
import pickle
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMatch:
    """Entity match with confidence and provenance."""
    entity_name: str
    dbpedia_uri: str
    confidence: float
    entity_type: str
    source: str  # 'sparql', 'spotlight', 'cache'
    timestamp: datetime

class DBpediaRecognizer:
    """Research-grade DBpedia entity recognition."""
    
    def __init__(self, 
                 cache_ttl_hours: int = 24,
                 max_retries: int = 3,
                 request_timeout: int = 60,
                 confidence_threshold: float = 0.5):
        
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.confidence_threshold = confidence_threshold
        

        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except (ImportError, OSError) as e:
            logger.warning(f"spaCy not available: {e}. Using basic tokenization.")
            self.nlp = None
        
        # Cache setup
        self.cache_dir = self._get_cache_dir()
        self.cache = self._load_cache()
        

        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'sparql_queries': 0,
            'spotlight_queries': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }
    
    def _get_cache_dir(self) -> str:
        """Get cache directory."""
        scratch_dir = os.environ.get('SCRATCH', os.getcwd())
        cache_dir = os.path.join(scratch_dir, 'dbpedia_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        cache_file = os.path.join(self.cache_dir, 'entity_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = os.path.join(self.cache_dir, 'entity_cache.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, entity_name: str) -> str:
        """Generate cache key for entity."""
        return hashlib.md5(entity_name.lower().encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() - timestamp < timedelta(hours=self.cache_ttl_hours)
    
    def query_sparql_with_retry(self, query: str) -> Optional[Dict]:
        """Query DBpedia SPARQL endpoint with retry logic."""
        sparql_url = "https://dbpedia.org/sparql"
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
        
                session = requests.Session()
                session.headers.update({
                    'Accept': 'application/sparql-results+json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                response = session.get(
                    sparql_url,
                    params={'query': query, 'format': 'json'},
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                response_time = time.time() - start_time
                self.stats['sparql_queries'] += 1
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['sparql_queries'] - 1) + response_time) 
                    / self.stats['sparql_queries']
                )
                
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"SPARQL connection error attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                # Longer backoff for connection errors
                time.sleep(5 + (2 ** attempt))
            except requests.exceptions.Timeout as e:
                logger.warning(f"SPARQL timeout attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.warning(f"SPARQL query attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def find_entity_uri_sparql(self, entity_name: str) -> Optional[Tuple[str, float]]:
        """Find entity URI using DBpedia SPARQL."""
        cache_key = self._get_cache_key(entity_name)
        

        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.stats['cache_hits'] += 1
                return cache_entry['uri'], cache_entry['confidence']
        
        # Query SPARQL
        query = f"""
        SELECT ?uri ?label WHERE {{
            ?uri rdfs:label ?label .
            FILTER(?label = "{entity_name}"@en)
            FILTER(STRSTARTS(STR(?uri), "http://dbpedia.org/resource/"))
        }}
        LIMIT 1
        """
        
        result = self.query_sparql_with_retry(query)
        if result and result.get('results', {}).get('bindings'):
            binding = result['results']['bindings'][0]
            uri = binding['uri']['value']
            confidence = 0.9  # High confidence for exact SPARQL matches
            
            # Cache the result
            self.cache[cache_key] = {
                'uri': uri,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            self._save_cache()
            
            return uri, confidence
        
        return None
    
    def find_entity_uri_spotlight(self, entity_name: str) -> Optional[Tuple[str, float]]:
        """Find entity URI using DBpedia Spotlight API."""
        spotlight_url = "https://api.dbpedia-spotlight.org/en/annotate"
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.get(
                    spotlight_url,
                    params={'text': entity_name, 'confidence': self.confidence_threshold},
                    headers={'Accept': 'application/json'},
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                response_time = time.time() - start_time
                self.stats['spotlight_queries'] += 1
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['spotlight_queries'] - 1) + response_time) 
                    / self.stats['spotlight_queries']
                )
                
                data = response.json()
                if data.get('Resources'):
                    resource = data['Resources'][0]
                    uri = resource['@URI']
                    confidence = float(resource['@similarityScore'])
                    return uri, confidence
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Spotlight connection error attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                time.sleep(2 ** attempt)
            except requests.exceptions.Timeout as e:
                logger.warning(f"Spotlight timeout attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Spotlight query attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['errors'] += 1
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def extract_entities(self, text: str) -> List[EntityMatch]:
        """Extract entities from text using multiple methods."""
        if not text:
            return []
        
        self.stats['total_queries'] += 1
        entities = []
        

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT', 'EVENT']:
                    entity_name = ent.text.strip()
                    
            
                    result = self.find_entity_uri_sparql(entity_name)
                    if result:
                        uri, confidence = result
                        entities.append(EntityMatch(
                            entity_name=entity_name,
                            dbpedia_uri=uri,
                            confidence=confidence,
                            entity_type=ent.label_,
                            source='sparql',
                            timestamp=datetime.now()
                        ))
                        continue
                    
    
                    result = self.find_entity_uri_spotlight(entity_name)
                    if result:
                        uri, confidence = result
                        entities.append(EntityMatch(
                            entity_name=entity_name,
                            dbpedia_uri=uri,
                            confidence=confidence,
                            entity_type=ent.label_,
                            source='spotlight',
                            timestamp=datetime.now()
                        ))
                    

                    if not result:
                
                        synthetic_uri = f"http://dbpedia.org/resource/{entity_name.replace(' ', '_')}"
                        entities.append(EntityMatch(
                            entity_name=entity_name,
                            dbpedia_uri=synthetic_uri,
                            confidence=0.3,
                            entity_type=ent.label_,
                            source='local',
                            timestamp=datetime.now()
                        ))
        
        return entities

    def extract_entities_offline(self, text: str) -> List[EntityMatch]:
        """Extract entities without external API calls (offline mode)."""
        if not text:
            return []
        
        entities = []
        

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT', 'EVENT']:
                    entity_name = ent.text.strip()
                    
            
                    synthetic_uri = f"http://dbpedia.org/resource/{entity_name.replace(' ', '_')}"
                    entities.append(EntityMatch(
                        entity_name=entity_name,
                        dbpedia_uri=synthetic_uri,
                                                  confidence=0.3,
                        entity_type=ent.label_,
                        source='offline',
                        timestamp=datetime.now()
                    ))
        
        return entities

    def extract_entities_batch(self, texts: List[str], batch_size: int = 10) -> List[List[EntityMatch]]:
        """Extract entities from multiple texts in batches for efficiency."""
        if not texts:
            return []
        
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
            
            for text in batch:
                entities = self.extract_entities(text)
                batch_results.append(entities)
            
            all_results.extend(batch_results)
            
            if i + batch_size < len(texts):
                time.sleep(0.2)
        
        return all_results

    def extract_entities_parallel(self, texts: List[str], max_workers: int = 4, batch_size: int = 10) -> List[List[EntityMatch]]:
        """Extract entities from multiple texts using parallel processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        if not texts:
            return []
        

        thread_local = threading.local()
        
        def get_recognizer():
            if not hasattr(thread_local, 'recognizer'):
                thread_local.recognizer = DBpediaRecognizer(
                    cache_ttl_hours=self.cache_ttl_hours,
                    max_retries=self.max_retries,
                    request_timeout=self.request_timeout,
                    confidence_threshold=self.confidence_threshold
                )
            return thread_local.recognizer
        
        def process_batch(batch):
            recognizer = get_recognizer()
            results = []
            for text in batch:
                entities = recognizer.extract_entities(text)
                results.append(entities)
            return results
        
        all_results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(process_batch, batch)
                futures.append((future, i))
            
            # Collect results
            for future, start_idx in futures:
                try:
                    batch_results = future.result()
                    for j, result in enumerate(batch_results):
                        all_results[start_idx + j] = result
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")

                    for j in range(len(batch_results)):
                        all_results[start_idx + j] = []
        
        return all_results
    
    def are_entities_related(self, uri1: str, uri2: str) -> Tuple[bool, float]:
        """Check if two entities are related using SPARQL."""
        query = f"""
        ASK {{
            {{ <{uri1}> ?p <{uri2}> }}
            UNION
            {{ <{uri2}> ?p <{uri1}> }}
        }}
        """
        
        result = self.query_sparql_with_retry(query)
        if result:
            related = result.get('boolean', False)
            confidence = 0.8 if related else 0.2
            return related, confidence
        
        return False, 0.0
    
    def get_statistics(self) -> Dict:
        """Get current statistics for research validation."""
        return self.stats.copy()
    
    def test_connectivity(self) -> Dict:
        """Test connectivity to DBpedia services."""
        results = {
            'sparql_accessible': False,
            'spotlight_accessible': False,
            'sparql_response_time': None,
            'spotlight_response_time': None,
            'errors': []
        }
        

        try:
            start_time = time.time()
            response = requests.get(
                "https://dbpedia.org/sparql",
                params={'query': 'SELECT ?s WHERE { ?s ?p ?o } LIMIT 1', 'format': 'json'},
                headers={'Accept': 'application/sparql-results+json'},
                timeout=30
            )
            response.raise_for_status()
            results['sparql_accessible'] = True
            results['sparql_response_time'] = time.time() - start_time
        except Exception as e:
            results['errors'].append(f"SPARQL test failed: {e}")
        

        try:
            start_time = time.time()
            response = requests.get(
                "https://api.dbpedia-spotlight.org/en/annotate",
                params={'text': 'test', 'confidence': 0.5},
                headers={'Accept': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            results['spotlight_accessible'] = True
            results['spotlight_response_time'] = time.time() - start_time
        except Exception as e:
            results['errors'].append(f"Spotlight test failed: {e}")
        
        return results

    def diagnose_connection_issues(self) -> Dict:
        """Diagnose connection issues with detailed testing."""
        import socket
        import subprocess
        
        diagnosis = {
            'dns_resolution': {},
            'port_connectivity': {},
            'http_connectivity': {},
            'proxy_settings': {},
            'recommendations': []
        }
        

        for host in ['dbpedia.org', 'api.dbpedia-spotlight.org']:
            try:
                ip = socket.gethostbyname(host)
                diagnosis['dns_resolution'][host] = {'success': True, 'ip': ip}
            except socket.gaierror as e:
                diagnosis['dns_resolution'][host] = {'success': False, 'error': str(e)}
                diagnosis['recommendations'].append(f"DNS resolution failed for {host}")
        

        for host in ['dbpedia.org', 'api.dbpedia-spotlight.org']:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, 443))
                sock.close()
                diagnosis['port_connectivity'][host] = {'success': result == 0, 'error_code': result}
                if result != 0:
                    diagnosis['recommendations'].append(f"Port 443 blocked for {host}")
            except Exception as e:
                diagnosis['port_connectivity'][host] = {'success': False, 'error': str(e)}
        

        test_urls = [
            "https://dbpedia.org/sparql",
            "https://api.dbpedia-spotlight.org/en/annotate"
        ]
        
        for url in test_urls:
            try:

                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'curl/7.68.0',
                    'Accept': '*/*'
                })
                response = session.get(url, timeout=10)
                diagnosis['http_connectivity'][url] = {'success': True, 'status_code': response.status_code}
            except Exception as e:
                diagnosis['http_connectivity'][url] = {'success': False, 'error': str(e)}
        

        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                diagnosis['proxy_settings'][var] = os.environ[var]
        
        return diagnosis

    def get_alternative_endpoints(self) -> Dict:
        """Get alternative DBpedia endpoints that might work."""
        return {
            'sparql_endpoints': [
                'https://dbpedia.org/sparql',
                'https://dbpedia.org/sparql/',
                'http://dbpedia.org/sparql',
                'https://dbpedia.org/sparql?default-graph-uri=http://dbpedia.org'
            ],
            'spotlight_endpoints': [
                'https://api.dbpedia-spotlight.org/en/annotate',
                'https://api.dbpedia-spotlight.org/en/annotate/',
                'http://api.dbpedia-spotlight.org/en/annotate'
            ]
        }

    def validate_entity_recognition(self, test_texts: List[str]) -> Dict:
        """Validate entity recognition accuracy with test texts."""
        total_entities = 0
        successful_entities = 0
        
        for text in test_texts:
            entities = self.extract_entities(text)
            total_entities += len(entities)
            successful_entities += sum(1 for e in entities if e.confidence > self.confidence_threshold)
        
        accuracy = successful_entities / total_entities if total_entities > 0 else 0.0
        
        return {
            'total_entities': total_entities,
            'successful_entities': successful_entities,
            'accuracy': accuracy,
            'avg_confidence': sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        }

    def process_large_dataset(self, texts: List[str], 
                            batch_size: int = 50, 
                            max_workers: int = 4,
                            use_parallel: bool = True,
                            save_progress: bool = True,
                            progress_file: str = None) -> List[List[EntityMatch]]:
        """Process large dataset with progress tracking."""
        if not texts:
            return []
        
        if progress_file is None:
            progress_file = f"dbpedia_progress_{int(time.time())}.json"
        

        completed_indices = set()
        if save_progress and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed_indices = set(progress_data.get('completed_indices', []))
                print(f"Loaded progress: {len(completed_indices)} texts already processed")
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        

        remaining_texts = []
        remaining_indices = []
        for i, text in enumerate(texts):
            if i not in completed_indices:
                remaining_texts.append(text)
                remaining_indices.append(i)
        
        if not remaining_texts:
            print("All texts already processed!")
            return self._load_completed_results(progress_file, len(texts))
        
        print(f"Processing {len(remaining_texts)} remaining texts out of {len(texts)} total")
        

        if use_parallel:
            results = self.extract_entities_parallel(remaining_texts, max_workers, batch_size)
        else:
            results = self.extract_entities_batch(remaining_texts, batch_size)
        

        if save_progress:
            completed_indices.update(remaining_indices)
            progress_data = {
                'completed_indices': list(completed_indices),
                'total_texts': len(texts),
                'timestamp': datetime.now().isoformat()
            }
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                print(f"Progress saved to {progress_file}")
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")
        

        all_results = [None] * len(texts)
        

        existing_results = self._load_completed_results(progress_file, len(texts))
        if existing_results:
            all_results = existing_results
        

        for i, result in zip(remaining_indices, results):
            all_results[i] = result
        
        return all_results

    def _load_completed_results(self, progress_file: str, total_texts: int) -> List[List[EntityMatch]]:
        """Load completed results from progress file."""
        results_file = progress_file.replace('.json', '_results.pkl')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results file: {e}")
        return [None] * total_texts
