"""
SatyaCheck AI - Main HTTP Handler
Complete and functional Cloud Function entry point for Google Gen AI Exchange Hackathon
"""

import functions_framework
import json
import logging
import time
import base64
from urllib.parse import urlparse
import requests
from datetime import datetime
from flask import Request, jsonify
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Local imports
from config import (
    GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID, GOOGLE_SEARCH_URL,
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, ERROR_MESSAGES, SUCCESS_MESSAGES,
    SOURCE_CREDIBILITY_SCORES, DOMAIN_EXPERTISE_BONUS, VALIDATION_RULES,
    FACT_CHECKING_SOURCES,
    is_production
)
from utils import (
    validate_input_text, validate_file_upload, validate_url, normalize_text,
    generate_claim_hash, generate_request_id, analyze_cultural_context,
    detect_manipulation_patterns, PerformanceTimer, get_current_timestamp,
    create_error_response, create_success_response, check_content_safety,
    detect_language_heuristic, extract_key_terms, extract_temporal_indicators,
    sanitize_error_message
)
from database import save_analysis_result, get_cached_analysis, get_system_health
from ai_analyzer import analyze_claim_with_ai, test_ai_analyzer
from content_processor import (
    get_content_processor, extract_text_from_any_content, validate_content_safety
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
content_processor = None
ai_analyzer = None

def initialize_services():
    """Initialize all services on first request"""
    global content_processor, ai_analyzer
    
    try:
        if content_processor is None:
            content_processor = get_content_processor()
            logger.info("Content processor initialized")
        
        if ai_analyzer is None:
            from ai_analyzer import get_ai_analyzer
            ai_analyzer = get_ai_analyzer()
            logger.info("AI analyzer initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False

# PIB FACT-CHECK INTEGRATION
def search_pib_factcheck(claim: str, max_results: int = 3) -> List[Dict]:
    """Search PIB fact-checks using Google Search API"""
    if not GOOGLE_SEARCH_API_KEY or not claim:
        return []
    
    try:
        with PerformanceTimer("pib_search"):
            key_terms = extract_key_terms(claim, max_terms=3)
            
            if not key_terms:
                return []
            
            from datetime import datetime
            current_year = datetime.now().year
            
            # ADVANCED GOOGLE DORKING for PIB search
            search_query = f'site:pib.gov.in (factcheck OR "fact check" OR verification) "{key_terms[0]}"'

            # Add additional relevant terms with OR operators
            if len(key_terms) > 1:
                search_query += f' ("{key_terms[1]}" OR {key_terms[1]})'

            # Add specific PIB fact-check keywords
            search_query += ' (fake OR misleading OR false OR true OR verified)'

            # If claim mentions current or future years, prioritize recent results
            if any(str(year) in claim for year in [current_year-1, current_year, current_year+1]):
                search_query += f" after:{current_year-1}-01-01"

            params = {
                'key': GOOGLE_SEARCH_API_KEY,
                'cx': GOOGLE_SEARCH_ENGINE_ID,
                'q': search_query,
                'num': max_results
            }
            
            response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Google Search API error: {response.status_code}")
                return []
            
            data = response.json()
            return parse_pib_search_results(data, claim)
            
    except Exception as e:
        logger.error(f"PIB search failed: {e}")
        return []
def search_web(claim: str, max_results: int = 5) -> List[Dict]:
    """Search web information using Google Search API"""
    if not GOOGLE_SEARCH_API_KEY or not claim:
        return []
    
    try:
        with PerformanceTimer("web_search"):
            key_terms = extract_key_terms(claim, max_terms=3)
            
            if not key_terms:
                return []
            
            # ADVANCED GOOGLE DORKING for web search with multiple strategies
            from datetime import datetime, timedelta
            recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            # Strategy 1: Official sources and fact-checkers
            primary_terms = key_terms[:2]
            fact_check_sites = [
                'factchecker.in', 'boomlive.in', 'altnews.in', 'thehindu.com',
                'indianexpress.com', 'reuters.com', 'pti.in'
            ]

            search_query = f'("{primary_terms[0]}") '

            # Add fact-check specific search
            search_query += f'(site:factchecker.in OR site:boomlive.in OR site:altnews.in) '
            search_query += f'(factcheck OR "fact check" OR verification OR debunk OR verify) '

            # Add second key term if available
            if len(primary_terms) > 1:
                search_query += f'("{primary_terms[1]}") '

            # Add temporal filter for recency
            search_query += f'after:{recent_date}'
            
            params = {
                'key': GOOGLE_SEARCH_API_KEY,
                'cx': GOOGLE_SEARCH_ENGINE_ID,
                'q': search_query,
                'num': max_results
            }
            
            response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Web search failed: {response.status_code}")
                return []
            
            data = response.json()
            return parse_web_search_results(data, claim)
            
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []

def parse_web_search_results(data: Dict, original_claim: str) -> List[Dict]:
    """Parse web search results for information"""
    results = []
    
    if 'items' not in data:
        return results
    
    for item in data['items'][:5]:
        try:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            
            domain = urlparse(link).netloc.lower()
            
            # Enhanced credibility scoring
            credibility_score = calculate_source_credibility(domain, original_claim)
            is_trusted = credibility_score >= 0.70  # Threshold for "trusted"
            
            relevance = calculate_claim_similarity(original_claim, f"{title} {snippet}")
            
            if relevance > 0.1:
                results.append({
                    'source': 'Web Search',
                    'title': title,
                    'snippet': snippet,
                    'url': link,
                    'domain': domain,
                    'is_trusted': is_trusted,
                    'relevance': relevance,
                    'confidence': credibility_score,  # Use calculated score instead of fixed values
                    'credibility_tier': 'high' if credibility_score >= 0.80 else 'medium' if credibility_score >= 0.60 else 'low'
                })
                
        except Exception as e:
            logger.error(f"Error parsing web result: {e}")
            continue
    
    return results

def parse_pib_search_results(data: Dict, original_claim: str) -> List[Dict]:
    """Parse Google Search results for PIB fact-checks"""
    results = []
    
    if 'items' not in data:
        return results
    
    for item in data['items'][:3]:
        try:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            
            content = f"{title} {snippet}".lower()
            verdict = extract_pib_verdict(content)
            relevance = calculate_claim_similarity(original_claim, content)
            
            # Enhanced PIB credibility
            domain = urlparse(link).netloc.lower()
            credibility_score = calculate_source_credibility(domain, original_claim)
            
            if relevance > 0.2:
                results.append({
                    'source': 'PIB Fact Check',
                    'title': title,
                    'snippet': snippet,
                    'url': link,
                    'verdict': verdict,
                    'relevance': relevance,
                    'confidence': credibility_score,
                    'credibility_tier': 'high'  # PIB is always high credibility
                })
                
        except Exception as e:
            logger.error(f"Error parsing PIB result: {e}")
            continue
    
    return results

def extract_pib_verdict(content: str) -> str:
    """Extract verdict from PIB content"""
    content_lower = content.lower()
    
    if any(pattern in content_lower for pattern in ['fact check', 'fake', 'false', 'misleading', 'incorrect']):
        return 'False'
    elif any(pattern in content_lower for pattern in ['true', 'correct', 'authentic', 'confirmed']):
        return 'True'
    elif any(pattern in content_lower for pattern in ['partially', 'misleading context', 'outdated']):
        return 'Partially True'
    else:
        return 'Unverified'

def calculate_claim_similarity(claim1: str, claim2: str) -> float:
    """Calculate simple similarity between claims"""
    terms1 = set(extract_key_terms(claim1.lower()))
    terms2 = set(extract_key_terms(claim2.lower()))
    
    if not terms1 or not terms2:
        return 0.0
    
    intersection = len(terms1.intersection(terms2))
    union = len(terms1.union(terms2))
    
    return intersection / union if union > 0 else 0.0

def calculate_source_credibility(domain: str, claim_text: str = "") -> float:
    """Calculate enhanced source credibility score"""
    # Get base score from domain
    base_score = SOURCE_CREDIBILITY_SCORES.get(domain, SOURCE_CREDIBILITY_SCORES['unknown'])
    
    # Add domain expertise bonus
    claim_lower = claim_text.lower()
    expertise_bonus = 0.0
    
    for topic, expert_domains in DOMAIN_EXPERTISE_BONUS.items():
        if any(expert_domain in domain for expert_domain in expert_domains):
            # Check if claim is related to this expertise area
            topic_keywords = {
                'health': ['health', 'medical', 'covid', 'vaccine', 'disease', 'treatment'],
                'politics': ['election', 'government', 'minister', 'parliament', 'policy'],
                'sports': ['cricket', 'football', 'olympics', 'tournament', 'match'],
                'technology': ['cyber', 'digital', 'tech', 'internet', 'software']
            }
            
            if any(keyword in claim_lower for keyword in topic_keywords.get(topic, [])):
                expertise_bonus = 0.05  # 5% bonus for domain expertise
                break
    
    # Apply recency bonus for recent content (if timestamp available)
    # This would need timestamp data from search results
    
    final_score = min(base_score + expertise_bonus, 0.98)  # Cap at 98%
    return final_score

def validate_and_filter_real_sources(search_results: List[Dict], claim_text: str) -> List[Dict]:
    """Validate and filter search results to return only real, accessible URLs"""
    verified_sources = []

    if not search_results:
        return verified_sources

    # Sort results by credibility and relevance
    sorted_results = sorted(
        search_results,
        key=lambda x: (x.get('confidence', 0) * x.get('relevance', 0)),
        reverse=True
    )

    for result in sorted_results[:4]:  # Maximum 4 sources
        url = result.get('url', '').strip()

        if not url or not url.startswith(('http://', 'https://')):
            continue

        # Validate URL format and accessibility
        try:
            parsed_url = urlparse(url)
            if not parsed_url.netloc:
                continue

            # Skip clearly fake or example URLs
            fake_indicators = [
                'example.com', 'test.com', 'sample.com', 'placeholder.com',
                'fake.com', 'dummy.com', 'mock.com', 'demo.com'
            ]

            if any(fake in parsed_url.netloc.lower() for fake in fake_indicators):
                continue

            # Validate domain credibility
            domain = parsed_url.netloc.lower()
            credibility_score = calculate_source_credibility(domain, claim_text)

            # Only include sources with reasonable credibility
            if credibility_score < 0.4:
                continue

            # Enhanced accessibility check with proper validation
            try:
                # Use HEAD request first, fallback to GET if needed
                response = requests.head(url, timeout=5, allow_redirects=True,
                                       headers={'User-Agent': 'SatyaCheck-Bot/1.0'})

                # If HEAD fails, try GET with limited content
                if response.status_code >= 400:
                    response = requests.get(url, timeout=5, allow_redirects=True,
                                          headers={'User-Agent': 'SatyaCheck-Bot/1.0'},
                                          stream=True)
                    response.raise_for_status()

                # Verify the final URL is not a redirect to an invalid page
                if response.url != url:
                    logger.info(f"URL redirected from {url} to {response.url}")
                    # Validate the final URL
                    final_parsed = urlparse(response.url)
                    if not final_parsed.netloc or any(fake in final_parsed.netloc.lower()
                                                    for fake in fake_indicators):
                        logger.warning(f"URL redirected to invalid destination: {response.url}")
                        continue

                if response.status_code >= 400:
                    logger.warning(f"URL not accessible: {url} (status: {response.status_code})")
                    continue

            except requests.exceptions.RequestException as e:
                logger.warning(f"URL accessibility check failed for {url}: {e}")
                continue

            # Check relevance to claim
            relevance = result.get('relevance', 0)
            if relevance < 0.2:  # Minimum relevance threshold
                continue

            # Prioritize official and fact-checking sources
            priority_domains = [
                'pib.gov.in', 'factchecker.in', 'boomlive.in', 'altnews.in',
                'thehindu.com', 'indianexpress.com', 'reuters.com', 'bbc.com'
            ]

            # Create proper source object
            source_obj = {
                "title": result.get('title', 'Search Result'),
                "url": url,
                "publisher": result.get('publisher', domain),
                "date": result.get('date', ''),
                "relevantQuote": result.get('snippet', ''),
                "credibilityScore": min(int(credibility_score * 10), 10)
            }

            # Add high-priority sources first
            if any(priority in domain for priority in priority_domains):
                if not any(s['url'] == url for s in verified_sources):
                    verified_sources.insert(0, source_obj)
            else:
                if not any(s['url'] == url for s in verified_sources):
                    verified_sources.append(source_obj)

        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            continue

    # Return maximum 3 verified sources
    final_sources = verified_sources[:3]

    logger.info(f"Verified {len(final_sources)} real sources from {len(search_results)} search results")

    return final_sources

# INPUT PROCESSING
def process_input_data(request: Request) -> Dict[str, Any]:
    """Process and validate input data from request"""
    input_data = {
        'type': 'text',
        'content': '',
        'language': DEFAULT_LANGUAGE,
        'metadata': {}
    }
    
    try:
        # Debug logging
        logger.info(f"Request method: {request.method}")
        logger.info(f"Form data keys: {list(request.form.keys())}")
        logger.info(f"Files keys: {list(request.files.keys())}")

        # Get language preference
        input_data['language'] = request.form.get('language', DEFAULT_LANGUAGE)
        if input_data['language'] not in SUPPORTED_LANGUAGES:
            input_data['language'] = DEFAULT_LANGUAGE

        # Process different input types
        if 'claim' in request.form:
            # Text input
            claim_text = request.form.get('claim', '').strip()
            is_valid, error_msg = validate_input_text(claim_text)
            
            if not is_valid:
                raise ValueError(error_msg)
            
            input_data['type'] = 'text'
            input_data['content'] = normalize_text(claim_text)
            
        elif 'image_data' in request.files:
            # Image input
            image_file = request.files['image_data']
            image_data = image_file.read()
            
            # Basic validation
            if not image_data or len(image_data) == 0:
                raise ValueError("Empty image file")

            if len(image_data) > VALIDATION_RULES['max_file_size']:
                raise ValueError("Image file too large")

            input_data['type'] = 'image'
            input_data['content'] = base64.b64encode(image_data).decode('utf-8')
            input_data['metadata'] = {
                'filename': image_file.filename or 'image.jpg',
                'size': len(image_data),
                'mime_type': image_file.content_type or 'image/jpeg'
            }
        
        elif 'audio_data' in request.files or 'live_audio_data' in request.form:
            # Smart audio source selection logic with better validation
            has_live_audio = 'live_audio_data' in request.form and request.form.get('live_audio_data', '').strip()
            has_uploaded_file = 'audio_data' in request.files and getattr(request.files['audio_data'], 'filename', None)

            if has_live_audio and has_uploaded_file:
                # Both sources available - prefer live recording (more recent)
                logger.info("Both live and uploaded audio found, using live recording")
                audio_source = "live"
            elif has_live_audio:
                audio_source = "live"
            elif has_uploaded_file:
                audio_source = "file"
            else:
                raise ValueError("No valid audio data provided")
            
            if audio_source == "live":
                # Live audio recording (base64 encoded)
                try:
                    audio_b64 = request.form.get('live_audio_data', '').strip()
                    if not audio_b64:
                        raise ValueError("Empty live audio data")
                    
                    audio_data = base64.b64decode(audio_b64)
                    
                    if len(audio_data) < 1000:
                        raise ValueError("Audio data too short")
                    
                    if len(audio_data) > 50 * 1024 * 1024:
                        raise ValueError("Audio data too large")
                    
                    input_data['type'] = 'audio'
                    input_data['content'] = audio_b64
                    input_data['metadata'] = {
                        'filename': 'live_recording.wav',
                        'size': len(audio_data),
                        'mime_type': 'audio/wav',
                        'language': f"{input_data['language']}-IN",
                        'audio_type': 'live',
                        'source_preference': 'live_recording'
                    }
                    
                except Exception as e:
                    raise ValueError(f"Invalid live audio data: {sanitize_error_message(str(e))}")
                    
            else:  # audio_source == "file"
                # Audio file upload
                audio_file = request.files['audio_data']
                audio_data = audio_file.read()

                # Basic validation
                if not audio_data or len(audio_data) == 0:
                    raise ValueError("Empty audio file")

                if len(audio_data) > VALIDATION_RULES['max_file_size']:
                    raise ValueError("Audio file too large")
                
                input_data['type'] = 'audio'
                input_data['content'] = base64.b64encode(audio_data).decode('utf-8')
                input_data['metadata'] = {
                    'filename': audio_file.filename or 'audio.wav',
                    'size': len(audio_data),
                    'mime_type': audio_file.content_type or 'audio/wav',
                    'language': f"{input_data['language']}-IN",
                    'audio_type': 'file',
                    'source_preference': 'uploaded_file'
                }

        elif 'video_data' in request.files:
            # Video file upload
            video_file = request.files['video_data']
            video_data = video_file.read()

            # Basic validation
            if not video_data or len(video_data) == 0:
                raise ValueError("Empty video file")

            if len(video_data) > VALIDATION_RULES['max_file_size']:
                raise ValueError("Video file too large")

            input_data['type'] = 'video'
            input_data['content'] = base64.b64encode(video_data).decode('utf-8')
            input_data['metadata'] = {
                'filename': video_file.filename or 'video.mp4',
                'size': len(video_data),
                'mime_type': video_file.content_type or 'video/mp4'
            }

        elif 'url' in request.form:
            # URL input
            url = request.form.get('url', '').strip()
            is_valid, error_msg = validate_url(url)
            
            if not is_valid:
                raise ValueError(error_msg)
            
            input_data['type'] = 'url'
            input_data['content'] = url
            
        else:
            # Debug detailed error message
            form_keys = list(request.form.keys())
            files_keys = list(request.files.keys())
            logger.error(f"No valid input found. Form keys: {form_keys}, Files keys: {files_keys}")

            error_details = []
            if not form_keys and not files_keys:
                error_details.append("No form data or files received")
            if form_keys:
                error_details.append(f"Form keys found: {form_keys}")
            if files_keys:
                error_details.append(f"File keys found: {files_keys}")

            raise ValueError(f"No valid input provided. Expected: 'claim', 'image_data', 'audio_data', 'live_audio_data', or 'url'. {'; '.join(error_details)}")
        
        # Add request metadata
        input_data['metadata'].update({
            'request_timestamp': get_current_timestamp(),
            'user_agent': request.headers.get('User-Agent', ''),
            'client_ip': request.environ.get('HTTP_X_FORWARDED_FOR', 
                                           request.environ.get('REMOTE_ADDR', 'unknown'))
        })
        
        return input_data
        
    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Request method: {request.method}")
        logger.error(f"Request content type: {request.content_type}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise ValueError(f"Input processing error: {sanitize_error_message(str(e))}")

# CONTENT EXTRACTION
def extract_content_from_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text content from various input types"""
    extraction_result = {
        'text': '',
        'confidence': 0.0,
        'metadata': {},
        'processing_time': 0.0
    }
    
    start_time = time.time()
    
    try:
        input_type = input_data['type']
        
        if input_type == 'text':
            extraction_result['text'] = input_data['content']
            extraction_result['confidence'] = 1.0
            
        elif input_type == 'image':
            image_data = base64.b64decode(input_data['content'])

            # Try to extract text, fallback on error
            try:
                text, confidence, analysis = extract_text_from_any_content(
                    image_data, 'image', input_data['metadata']
                )
                extraction_result['text'] = text
                extraction_result['confidence'] = confidence
                extraction_result['metadata'].update(analysis)
            except Exception as extract_error:
                logger.warning(f"Image text extraction failed: {extract_error}")
                # Fallback: provide instruction for manual analysis
                extraction_result['text'] = "Please describe what you see in this image for fact-checking analysis."
                extraction_result['confidence'] = 0.5
                extraction_result['metadata'] = {
                    'extraction_method': 'fallback',
                    'original_error': sanitize_error_message(str(extract_error)),
                    'requires_manual_description': True
                }
            
        elif input_type == 'audio':
            audio_data = base64.b64decode(input_data['content'])

            # Try to extract text, fallback on error
            try:
                text, confidence, analysis = extract_text_from_any_content(
                    audio_data, 'audio', input_data['metadata']
                )
                extraction_result['text'] = text
                extraction_result['confidence'] = confidence
                extraction_result['metadata'].update(analysis)
            except Exception as extract_error:
                logger.warning(f"Audio text extraction failed: {extract_error}")
                # Fallback: provide instruction for manual analysis
                extraction_result['text'] = "Please type what was said in the audio for fact-checking analysis."
                extraction_result['confidence'] = 0.5
                extraction_result['metadata'] = {
                    'extraction_method': 'fallback',
                    'original_error': sanitize_error_message(str(extract_error)),
                    'requires_manual_transcription': True,
                    'audio_size': len(audio_data)
                }

        elif input_type == 'video':
            video_data = base64.b64decode(input_data['content'])

            # Videos always use deepfake detection (no text extraction from video)
            logger.info("Video input detected - running deepfake detection")
            try:
                import tempfile
                import os
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                        temp_file.write(video_data)
                        temp_path = temp_file.name

                    from deepfake_detection import analyze_media_for_deepfake
                    deepfake_result = analyze_media_for_deepfake(temp_path, 'video')

                    extraction_result['deepfake_analysis'] = deepfake_result
                    extraction_result['text'] = ""  # Empty to trigger media-only mode
                    extraction_result['metadata']['analysis_mode'] = 'video_deepfake_only'
                    extraction_result['metadata']['skip_text_analysis'] = True
                    logger.info(f"Video deepfake analysis complete: {deepfake_result.get('is_deepfake', 'unknown')}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as deepfake_error:
                logger.warning(f"Video deepfake detection failed: {deepfake_error}")
                extraction_result['deepfake_analysis'] = {
                    'error': 'failed',
                    'message': sanitize_error_message(str(deepfake_error))
                }
                extraction_result['text'] = ""
                extraction_result['metadata']['analysis_mode'] = 'video_deepfake_only'
                extraction_result['metadata']['skip_text_analysis'] = True

        elif input_type == 'url':
            url = input_data['content']

            # Try to extract text, fallback on error
            try:
                text, confidence, analysis = extract_text_from_any_content(
                    url, 'url', input_data['metadata']
                )
                extraction_result['text'] = text
                extraction_result['confidence'] = confidence
                extraction_result['metadata'].update(analysis)
            except Exception as extract_error:
                logger.warning(f"URL text extraction failed: {extract_error}")
                # Fallback: provide instruction for manual analysis
                extraction_result['text'] = f"Please copy and paste the content from this URL for analysis: {url}"
                extraction_result['confidence'] = 0.5
                extraction_result['metadata'] = {
                    'extraction_method': 'fallback',
                    'original_error': sanitize_error_message(str(extract_error)),
                    'requires_manual_copy': True,
                    'original_url': url
                }
            
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        extraction_result['processing_time'] = time.time() - start_time

        # Check if this is media-only mode (deepfake detection without text analysis)
        is_media_only = extraction_result['metadata'].get('skip_text_analysis', False)

        # For media-only mode, empty text is expected - skip validation
        if not is_media_only:
            # For fallback cases, allow the instruction text through
            if (not extraction_result['text'] or
                (len(extraction_result['text'].strip()) < 5 and
                 extraction_result['metadata'].get('extraction_method') != 'fallback')):
                raise ValueError("No meaningful content could be extracted from the input")

        logger.info(f"Content extraction completed: {len(extraction_result['text'])} characters (media-only: {is_media_only})")

        return extraction_result
        
    except Exception as e:
        extraction_result['processing_time'] = time.time() - start_time
        logger.error(f"Content extraction failed: {e}")
        raise ValueError(f"Content extraction error: {sanitize_error_message(str(e))}")

# SOURCE URL SUPPORT
def get_comprehensive_fact_checking_sources() -> List[Dict]:
    """
    Get curated list of top fact-checking sources.
    Returns most relevant Indian and international fact-checking platforms.
    """
    all_fact_checkers = []

    # Add top 3 Indian fact-checkers
    for source in FACT_CHECKING_SOURCES['indian'][:3]:
        source_with_type = source.copy()
        source_with_type['sourceType'] = 'fact_checker'
        source_with_type['region'] = 'Indian'
        all_fact_checkers.append(source_with_type)

    # Add top 2 international fact-checkers
    for source in FACT_CHECKING_SOURCES['international'][:2]:
        source_with_type = source.copy()
        source_with_type['sourceType'] = 'fact_checker'
        source_with_type['region'] = 'International'
        all_fact_checkers.append(source_with_type)

    return all_fact_checkers

# COMPREHENSIVE ANALYSIS PIPELINE
def analyze_claim_comprehensive(extracted_content: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive fact-checking analysis pipeline"""
    analysis_start_time = time.time()
    
    try:
        claim_text = extracted_content['text']
        language = input_data['language']

        claim_hash = generate_claim_hash(claim_text)
        claim_id = generate_request_id()

        logger.info(f"Starting analysis for claim: {claim_id}")

        # FAST TRACK: Check for obvious fake patterns first (instant response)
        obvious_fake_check = check_obvious_fake_patterns_cached(claim_text)
        if obvious_fake_check['is_obvious_fake']:
            logger.info(f"INSTANT VERDICT: Obvious fake detected with {obvious_fake_check['confidence']}% confidence")
            return create_instant_fake_response(
                claim_text, obvious_fake_check, claim_id, language,
                time.time() - analysis_start_time
            )

        # Check cache for complex analysis
        cached_result = get_cached_analysis(claim_hash)
        if cached_result and not is_production():
            logger.info("Returning cached analysis result")
            cached_result['analysis_result']['claim_id'] = claim_id
            cached_result['analysis_result']['processing_time'] = time.time() - analysis_start_time
            return cached_result['analysis_result']
        
        # Content safety check
        safety_check = check_content_safety(claim_text)
        if not safety_check['is_safe'] and safety_check['risk_level'] == 'high':
            return {
                'verdict': 'Content Flagged',
                'score': 0,
                'explanation': 'This content contains potentially harmful material that cannot be analyzed.',
                'explainLikeAKid': 'This content might not be safe to share.',
                'sources': [],
                'categories': {},
                'processing_time': time.time() - analysis_start_time,
                'content_safety': safety_check,
                'claim_id': claim_id,
                'language': language
            }
        
        # Cultural context analysis
        cultural_context = analyze_cultural_context(claim_text)
        
        # Add temporal context analysis
        temporal_context = extract_temporal_indicators(claim_text)
        cultural_context['temporal_context'] = temporal_context
        # PIB search in parallel
# PIB and web search in parallel
        pib_results = []
        web_results = []
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                pib_future = executor.submit(search_pib_factcheck, claim_text, 3)
                web_future = executor.submit(search_web, claim_text, 5)
                
                try:
                    pib_results = pib_future.result(timeout=8.0)
                except TimeoutError:
                    logger.warning("PIB search timeout")
                    pib_results = []
                
                try:
                    web_results = web_future.result(timeout=8.0)
                except TimeoutError:
                    logger.warning("Web search timeout")
                    web_results = []
                
        except Exception as e:
            logger.warning(f"PIB search error: {e}")
            pib_results = []
        
        # AI analysis with context
        # Combine all search results
        all_search_results = pib_results + web_results
        
        # AI analysis with enhanced context
        ai_analysis_result = analyze_claim_with_ai(
            claim_text,
            cultural_context,
            all_search_results,
            language
        )
        
        # Add REAL verified sources only (no hypothetical URLs)
        verified_sources = validate_and_filter_real_sources(all_search_results, claim_text)

        # Enhanced fallback sources - only add well-known, verified fact-checking sites
        if not verified_sources:
            # These are verified, accessible fact-checking websites
            default_sources = [
                {
                    "title": "PIB Fact Check",
                    "url": "https://pib.gov.in/factcheck.aspx",
                    "publisher": "Press Information Bureau",
                    "date": "",
                    "relevantQuote": "Official government fact-checking portal",
                    "credibilityScore": 10
                },
                {
                    "title": "FactChecker.in",
                    "url": "https://www.factchecker.in/",
                    "publisher": "FactChecker.in",
                    "date": "",
                    "relevantQuote": "India's first data-driven fact-checking initiative",
                    "credibilityScore": 9
                }
            ]
            # Verify these default sources are actually accessible before adding
            verified_defaults = []
            for default_source in default_sources:
                try:
                    response = requests.head(default_source["url"], timeout=3,
                                           headers={'User-Agent': 'SatyaCheck-Bot/1.0'})
                    if response.status_code < 400:
                        verified_defaults.append(default_source)
                except:
                    logger.warning(f"Default source not accessible: {default_source['url']}")
                    continue

            verified_sources = verified_defaults[:2]  # Maximum 2 verified default sources

        ai_analysis_result['sources'] = verified_sources

        # Get comprehensive fact-checking sources (clean output - only fact-checking platforms)
        fact_checking_sources = get_comprehensive_fact_checking_sources()
        ai_analysis_result['fact_checking_sources'] = fact_checking_sources

        logger.info(f"Sources summary - Specific: {len(verified_sources)}, Fact-checkers: {len(fact_checking_sources)}")

        # Add metadata
        ai_analysis_result.update({
            'claim_id': claim_id,
            'claim_hash': claim_hash,
            'processing_time': time.time() - analysis_start_time,
            'analysis_timestamp': get_current_timestamp(),
            'input_type': input_data['type'],
            'language': language,
            'content_safety': safety_check
        })
        
        # Store result
        save_analysis_result(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_hash=claim_hash,
            language=language,
            verdict=ai_analysis_result.get('verdict', 'Error'),
            confidence_score=ai_analysis_result.get('score', 0),
            analysis_result=ai_analysis_result,
            processing_time=ai_analysis_result['processing_time']
        )
        
        logger.info(f"Analysis completed: verdict={ai_analysis_result.get('verdict')}, score={ai_analysis_result.get('score')}")
        
        return ai_analysis_result
        
    except Exception as e:
        processing_time = time.time() - analysis_start_time
        logger.error(f"Comprehensive analysis failed: {e}")
        
        return {
            'verdict': 'Error',
            'score': 0,
            'explanation': f'Analysis failed due to technical error: {sanitize_error_message(str(e))}',
            'explainLikeAKid': 'Something went wrong while checking this information. Please try again.',
            'sources': [],
            'categories': {},
            'processing_time': processing_time,
            'error': sanitize_error_message(str(e)),
            'analysis_timestamp': get_current_timestamp(),
            'claim_id': generate_request_id(),
            'language': input_data.get('language', DEFAULT_LANGUAGE)
        }

# PERFORMANCE OPTIMIZATIONS AND CACHING

def check_obvious_fake_patterns_cached(claim_text: str) -> Dict[str, Any]:
    """Check for obvious fake patterns with caching for instant response"""
    from utils import cache, generate_claim_hash

    # Cache key for obvious fake patterns
    pattern_hash = generate_claim_hash(f"obvious_fake_{claim_text}")
    cached_result = cache.get(pattern_hash)

    if cached_result:
        return cached_result

    # Perform obvious fake detection
    result = detect_obvious_fake_patterns_optimized(claim_text)

    # Cache the result for 24 hours if it's obviously fake
    if result['is_obvious_fake']:
        cache.set(pattern_hash, result, ttl=86400)  # 24 hours cache

    return result

def detect_obvious_fake_patterns_optimized(claim_text: str) -> Dict[str, Any]:
    """Optimized obvious fake detection for instant response"""
    claim_lower = claim_text.lower()
    confidence = 0
    patterns = []
    reasoning = []

    # SUPER FAST PATTERNS - Most common Indian fake news indicators
    instant_fake_patterns = {
        'viral_forwards': {
            'patterns': ['forward to 10', 'govt will delete', 'share with everyone', 'viral video', 'dont ignore'],
            'weight': 40
        },
        'impossible_amounts': {
            'patterns': [r'₹\s*([5-9]\d{4,})', r'₹\s*([1-9]\d{5,})'],  # 50k+ amounts
            'weight': 45
        },
        'miracle_claims': {
            'patterns': ['miracle cure', 'doctors hate', 'secret that', 'free 5g', 'free wifi everywhere'],
            'weight': 50
        },
        'urgent_without_source': {
            'patterns': ['urgent breaking', 'government announcement today', 'last chance'],
            'weight': 35
        }
    }

    for category, data in instant_fake_patterns.items():
        for pattern in data['patterns']:
            if pattern in claim_lower:
                patterns.append(category)
                reasoning.append(f'Contains {category} indicator: "{pattern}"')
                confidence += data['weight']
                break  # Only count once per category

    return {
        'is_obvious_fake': confidence >= 70,
        'confidence': min(confidence, 95),
        'patterns': patterns,
        'reasoning': '; '.join(reasoning) if reasoning else 'No obvious fake patterns detected',
        'detection_time': 'instant'
    }

def create_instant_fake_response(claim_text: str, fake_check: Dict, claim_id: str,
                                language: str, processing_time: float) -> Dict[str, Any]:
    """Create instant response for obvious fake claims"""

    language_names = {'hi': 'Hindi', 'en': 'English', 'ta': 'Tamil', 'te': 'Telugu',
                      'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati'}
    lang_name = language_names.get(language, 'English')

    # Create response in appropriate language
    if language == 'hi':
        explanation = f"यह दावा स्पष्ट रूप से गलत है। {fake_check['reasoning']} जैसे संकेतक मिले हैं जो आमतौर पर फर्जी खबरों में पाए जाते हैं।"
        kid_explanation = "यह जानकारी सच नहीं है। इसे शेयर न करें।"
        categories_hindi = {
            'source_credibility': {
                'label': 'स्रोत की विश्वसनीयता',
                'confidence': fake_check['confidence'],
                'short': 'विश्वसनीय स्रोत नहीं मिला',
                'medium': 'इस दावे में विश्वसनीय स्रोत का अभाव है और फर्जी खबर के संकेतक मौजूद हैं।',
                'long': 'यह दावा विश्वसनीय स्रोतों की कमी और फर्जी खबरों के सामान्य पैटर्न दिखाता है।',
                'action': 'आधिकारिक स्रोतों से सत्यापन करें'
            }
        }
        categories = categories_hindi
    else:
        explanation = f"This claim is clearly false. {fake_check['reasoning']} These are common indicators found in fake news."
        kid_explanation = "This information is not true. Don't share it."
        categories = {
            'source_credibility': {
                'label': 'Source Trustworthiness',
                'confidence': fake_check['confidence'],
                'short': 'No reliable sources found',
                'medium': 'This claim lacks credible sources and shows typical fake news indicators.',
                'long': 'The claim demonstrates absence of reliable sources and common fake news patterns.',
                'action': 'Verify with official sources before sharing'
            }
        }

    return {
        'verdict': 'False',
        'score': fake_check['confidence'],
        'explanation': explanation,
        'explainLikeAKid': kid_explanation,
        'sources': [],  # No sources for obvious fakes
        'categories': categories,
        'claim_id': claim_id,
        'processing_time': processing_time,
        'analysis_timestamp': get_current_timestamp(),
        'language': language,
        'fast_track': True,
        'detection_method': 'obvious_fake_patterns',
        'pattern_confidence': fake_check['confidence']
    }

# CORS HELPER
def add_cors_headers(response):
    """Add CORS headers to response"""
    if hasattr(response, 'headers'):
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Max-Age", "3600")
    return response

# MAIN HTTP ENDPOINT WITH ROUTING
@functions_framework.http
def satyacheck_main(request: Request):
    """Main SatyaCheck AI endpoint with proper routing"""

    # Handle CORS preflight for all routes
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Route based on path
    path = request.path.lower().strip('/')

    if path == 'health' or path == '':
        if request.method == 'GET':
            return handle_health_check(request)

    if path == 'demo':
        return handle_demo_endpoint(request)

    if path == 'debug':
        return handle_debug_request(request)

    # Default to analyze endpoint for POST requests
    if request.method == 'POST':
        return handle_analyze_request(request)

    # Invalid route or method
    return add_cors_headers(jsonify(create_error_response(
        'invalid_endpoint',
        f'Invalid endpoint or method: {request.method} {request.path}'
    ))), 404

def handle_analyze_request(request: Request):
    """Handle analyze requests - main fact-checking logic"""

    request_start_time = time.time()

    try:
        # Initialize services
        if not initialize_services():
            return add_cors_headers(jsonify(create_error_response(
                'service_unavailable',
                'Service initialization failed. Please try again later.'
            ))), 503

        # Process input
        try:
            input_data = process_input_data(request)
            logger.info(f"Processing {input_data['type']} input in {input_data['language']}")
        except ValueError as e:
            return add_cors_headers(jsonify(create_error_response(
                'invalid_input',
                sanitize_error_message(str(e))
            ))), 400

        # Extract content
        try:
            extracted_content = extract_content_from_input(input_data)
            logger.info(f"Content extracted: {len(extracted_content['text'])} chars")
        except ValueError as e:
            return add_cors_headers(jsonify(create_error_response(
                'content_extraction_failed',
                sanitize_error_message(str(e))
            ))), 400

        # Check if this is media-only deepfake analysis (skip claim analysis)
        analysis_mode = extracted_content.get('metadata', {}).get('analysis_mode', '')
        is_media_deepfake_only = analysis_mode in ['image_deepfake_only', 'audio_deepfake_only', 'video_deepfake_only']

        if is_media_deepfake_only:
            # Media-only mode: Return ONLY deepfake results, NO claim analysis
            media_type = "Video" if analysis_mode == 'video_deepfake_only' else ("Audio" if analysis_mode == 'audio_deepfake_only' else "Image")
            logger.info(f"{media_type} deepfake-only mode detected - returning deepfake analysis only")

            deepfake_data = extracted_content.get('deepfake_analysis', {})

            # Create minimal response with ONLY deepfake data
            analysis_result = {
                'deepfake_analysis': deepfake_data,
                'media_only_mode': True,
                'analysis_mode': analysis_mode,
                'claim_id': generate_request_id(),
                'language': input_data['language'],
                'metadata': extracted_content.get('metadata', {}),
                'processing_time': round(time.time() - request_start_time, 2)
            }

            # Add mode-specific message
            if analysis_mode == 'image_deepfake_only':
                analysis_result['message'] = 'Image analyzed for authenticity and manipulation.'
            elif analysis_mode == 'audio_deepfake_only':
                analysis_result['message'] = 'Audio analyzed for voice cloning and deepfake detection.'
            elif analysis_mode == 'video_deepfake_only':
                analysis_result['message'] = 'Video analyzed for authenticity and deepfake detection.'
        else:
            # Normal mode: Analyze claim
            analysis_result = analyze_claim_comprehensive(extracted_content, input_data)

        # Add metadata
        total_processing_time = time.time() - request_start_time
        analysis_result.update({
            'total_processing_time': round(total_processing_time, 2),
            'extraction_metadata': {
                'confidence': extracted_content['confidence'],
                'extraction_time': extracted_content['processing_time']
            }
        })

        logger.info(f"Request completed in {total_processing_time:.2f}s")

        # Generate audio if requested
        include_audio = request.form.get('include_audio', 'false').lower() == 'true'
        if include_audio and content_processor:
            try:
                explanation_text = analysis_result.get('explanation', '')
                if explanation_text:
                    audio_response = content_processor.generate_audio_response(
                        explanation_text,
                        SUPPORTED_LANGUAGES[input_data['language']]['tts_code']
                    )
                    if audio_response:
                        analysis_result['audio_content'] = audio_response
            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")

        return add_cors_headers(jsonify(create_success_response(
            analysis_result,
            SUCCESS_MESSAGES['analysis_complete']
        )))

    except Exception as e:
        total_processing_time = time.time() - request_start_time
        logger.error(f"Request failed after {total_processing_time:.2f}s: {e}")

        return add_cors_headers(jsonify(create_error_response(
            'internal_error',
            ERROR_MESSAGES['processing_failed'],
            {'processing_time': round(total_processing_time, 2)}
        ))), 500

def handle_health_check(request: Request):
    """Handle health check requests"""
    try:
        initialize_services()

        health_status = {
            'status': 'healthy',
            'timestamp': get_current_timestamp(),
            'version': '2.0.0',
            'environment': 'production' if is_production() else 'development'
        }

        services_health = {}
        overall_healthy = True

        # Check services
        try:
            db_health = get_system_health()
            services_health['database'] = db_health.get('status', 'unknown')
            if db_health.get('status') != 'healthy':
                overall_healthy = False
        except Exception as e:
            services_health['database'] = f'error: {sanitize_error_message(str(e))}'
            overall_healthy = False

        try:
            ai_test = test_ai_analyzer()
            services_health['ai_analyzer'] = 'healthy' if ai_test else 'degraded'
            if not ai_test:
                overall_healthy = False
        except Exception as e:
            services_health['ai_analyzer'] = f'error: {sanitize_error_message(str(e))}'
            overall_healthy = False

        try:
            if content_processor:
                processor_health = content_processor.get_processor_health()
                services_health['content_processor'] = processor_health.get('overall', 'unknown')
            else:
                services_health['content_processor'] = 'not_initialized'
                overall_healthy = False
        except Exception as e:
            services_health['content_processor'] = f'error: {sanitize_error_message(str(e))}'
            overall_healthy = False

        try:
            if GOOGLE_SEARCH_API_KEY:
                services_health['pib_search'] = 'healthy'
            else:
                services_health['pib_search'] = 'api_key_missing'
        except Exception as e:
            services_health['pib_search'] = f'error: {sanitize_error_message(str(e))}'

        health_status['services'] = services_health
        health_status['status'] = 'healthy' if overall_healthy else 'degraded'

        status_code = 200 if overall_healthy else 503
        return add_cors_headers(jsonify(health_status)), status_code

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return add_cors_headers(jsonify({
            'status': 'unhealthy',
            'timestamp': get_current_timestamp(),
            'error': sanitize_error_message(str(e))
        })), 500

def handle_demo_endpoint(request: Request):
    """Handle demo endpoint requests"""
    try:
        sample_claims = [
            {
                'claim': "गाय का मूत्र पीने से कोरोना वायरस ठीक हो जाता है",
                'language': 'hi',
                'expected_verdict': 'False'
            },
            {
                'claim': "Scientists discovered this miracle cure that doctors don't want you to know about",
                'language': 'en',
                'expected_verdict': 'False'
            },
            {
                'claim': "PM announces ₹5000 monthly scheme for all Indian families",
                'language': 'en',
                'expected_verdict': 'Unverified'
            }
        ]

        if not initialize_services():
            return add_cors_headers(jsonify({
                'error': 'Service initialization failed',
                'sample_claims': sample_claims
            })), 503

        results = []
        for sample in sample_claims:
            try:
                demo_result = {
                    'claim': sample['claim'],
                    'verdict': sample['expected_verdict'],
                    'score': 85 if sample['expected_verdict'] == 'False' else 30,
                    'explanation': f"Based on cultural context analysis and fact-checking, this claim appears to be {sample['expected_verdict'].lower()}.",
                    'explainLikeAKid': "This information might not be true. Always check with trusted sources.",
                    'categories': {
                        'source_trust': {
                            'label': 'Needs Verification',
                            'confidence': 75,
                            'short': 'Sources need verification',
                            'action': 'Check official sources'
                        }
                    },
                    'processing_time': 0.5,
                    'language': sample['language'],
                    'claim_id': generate_request_id()
                }
                results.append(demo_result)

            except Exception as e:
                logger.error(f"Demo processing failed for {sample['claim']}: {e}")
                results.append({
                    'claim': sample['claim'],
                    'error': sanitize_error_message(str(e)),
                    'language': sample['language']
                })

        return add_cors_headers(jsonify({
            'success': True,
            'demo_results': results,
            'message': 'SatyaCheck AI Demo - Sample fact-checking results',
            'timestamp': get_current_timestamp()
        }))

    except Exception as e:
        logger.error(f"Demo endpoint failed: {e}")
        return add_cors_headers(jsonify({
            'success': False,
            'error': sanitize_error_message(str(e)),
            'message': 'Demo endpoint error'
        })), 500

def handle_debug_request(request: Request):
    """Handle debug requests"""
    try:
        debug_info = {
            'method': request.method,
            'content_type': request.content_type,
            'form_keys': list(request.form.keys()),
            'files_keys': list(request.files.keys()),
            'form_data': {key: str(value)[:100] for key, value in request.form.items()},  # Truncate values
            'headers': dict(request.headers),
            'url': request.url,
            'endpoint': request.endpoint,
            'path': request.path
        }

        response = jsonify(create_success_response(debug_info, 'Debug info collected'))
        return add_cors_headers(response)

    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}")
        response = jsonify(create_error_response('debug_failed', sanitize_error_message(str(e))))
        return add_cors_headers(response), 500


logger.info("SatyaCheck AI Main Module Loaded Successfully")
logger.info(f"Environment: {'Production' if is_production() else 'Development'}")
logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
logger.info("🚀 SatyaCheck AI Backend v2.0.0 Ready for Production")
logger.info("🧠 Advanced AI-powered fact-checking with cultural context")
logger.info("🇮🇳 Optimized for Indian misinformation patterns")
logger.info("🌐 Multi-language and multi-modal support active")
logger.info("🎤 Live audio recording and file upload support enabled")
logger.info("📊 PIB integration and analytics enabled")