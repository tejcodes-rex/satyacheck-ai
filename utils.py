"""
SatyaCheck AI - Utility Functions
Complete utilities for text processing, validation, and common operations
"""

import re
import hashlib
import json
import logging
import time
import mimetypes
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlparse
import base64

from config import (
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, CULTURAL_PATTERNS,
    MANIPULATION_INDICATORS, VALIDATION_RULES, ERROR_MESSAGES,
    HARMFUL_CONTENT_PATTERNS, LOG_CONFIG
)

import requests

def sanitize_error_message(error_message: str) -> str:
    """Sanitize error messages to remove sensitive information like API keys"""
    if not error_message:
        return error_message

    # Remove API keys from URLs (pattern: key=AIza...)
    error_message = re.sub(r'key=[A-Za-z0-9_-]{20,}', 'key=***REDACTED***', error_message)

    # Remove API keys in general format (AIza...)
    error_message = re.sub(r'AIza[A-Za-z0-9_-]{32,}', '***REDACTED***', error_message)

    # Remove other common API key patterns
    error_message = re.sub(r'["\']?[A-Za-z0-9_-]{32,}["\']?', lambda m: '***REDACTED***' if 'key' in error_message.lower() else m.group(0), error_message)

    return error_message

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format']
)
logger = logging.getLogger(__name__)

# === TEXT PROCESSING UTILITIES ===

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    if not text:
        return ""
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def clean_text(text: str) -> str:
    """Clean text while preserving important punctuation"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers (Indian format)
    text = re.sub(r'[\+]?[91]?[6-9]\d{9}', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`´]', '"', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return normalize_text(text)

def extract_key_terms(text: str, max_terms: int = 10) -> List[str]:
    """Extract key terms from text for search and analysis"""
    if not text:
        return []
    
    # Common stop words in multiple languages
    stop_words = {
        'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
               'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
               'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'},
        'hi': {'है', 'के', 'में', 'से', 'को', 'का', 'की', 'पर', 'हैं', 'थे', 'गया',
               'कर', 'कि', 'या', 'भी', 'एक', 'यह', 'वह', 'जो', 'तो', 'अब', 'और'}
    }
    
    # Clean and normalize text
    text = clean_text(text.lower())
    
    # Extract words (including Devanagari and other Indian scripts)
    words = re.findall(r'[\w\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]+', text)
    
    # Filter out stop words and short words
    filtered_words = []
    for word in words:
        if len(word) >= 3:
            # Check if word is not a stop word in any language
            is_stop_word = False
            for lang_stop_words in stop_words.values():
                if word in lang_stop_words:
                    is_stop_word = True
                    break
            if not is_stop_word:
                filtered_words.append(word)
    
    # Count frequency and return top terms
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top terms
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_terms]]

def detect_language_heuristic(text: str) -> str:
    """Detect language using heuristic pattern matching"""
    if not text:
        return DEFAULT_LANGUAGE
    
    text_lower = text.lower()
    
    # Check for non-Latin scripts first
    if re.search(r'[\u0900-\u097F]', text):  # Devanagari (Hindi)
        return 'hi'
    elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
        return 'ta'
    elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
        return 'te'
    elif re.search(r'[\u0980-\u09FF]', text):  # Bengali
        return 'bn'
    elif re.search(r'[\u0D80-\u0DFF]', text):  # Marathi
        return 'mr'
    elif re.search(r'[\u0A80-\u0AFF]', text):  # Gujarati
        return 'gu'
    
    # Fallback to English if mostly Latin script
    return 'en' if re.search(r'[a-zA-Z]', text) else DEFAULT_LANGUAGE

# === VALIDATION UTILITIES ===

def validate_input_text(text: str) -> Tuple[bool, Optional[str]]:
    """Validate input text against rules"""
    if not text or not text.strip():
        return False, ERROR_MESSAGES['invalid_input']
    
    text = text.strip()
    
    if len(text) < VALIDATION_RULES['min_claim_length']:
        return False, ERROR_MESSAGES['claim_too_short']
    
    if len(text) > VALIDATION_RULES['max_claim_length']:
        return False, ERROR_MESSAGES['claim_too_long']
    
    # Check for suspicious patterns (excessive special characters)
    special_char_ratio = len(re.findall(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]', text)) / len(text)
    if special_char_ratio > 0.3:  # More than 30% special characters
        return False, "Text contains too many special characters"
    
    return True, None

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL format and accessibility"""
    if not url:
        return False, ERROR_MESSAGES['invalid_input']
    
    if len(url) > VALIDATION_RULES['max_url_length']:
        return False, "URL too long"
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"
        
        if parsed.scheme not in ['http', 'https']:
            return False, "Only HTTP and HTTPS URLs are allowed"
        
        return True, None
    except Exception:
        return False, "Invalid URL format"

def validate_file_upload(file_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file"""
    if not file_data:
        return False, ERROR_MESSAGES['invalid_input']
    
    # Check file size
    if len(file_data) > VALIDATION_RULES['max_file_size']:
        return False, ERROR_MESSAGES['file_too_large']
    
    # Check file extension
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    allowed_formats = (VALIDATION_RULES['allowed_image_formats'] + 
                      VALIDATION_RULES['allowed_audio_formats'])
    
    if file_extension not in allowed_formats:
        return False, ERROR_MESSAGES['unsupported_format']
    
    # Basic file header validation
    if not _validate_file_header(file_data, file_extension):
        return False, "File header doesn't match extension"
    
    return True, None

def _validate_file_header(file_data: bytes, extension: str) -> bool:
    """Validate file header matches extension"""
    if len(file_data) < 8:
        return False
    
    header = file_data[:8]
    
    # Common file signatures
    signatures = {
        'jpg': [b'\xFF\xD8\xFF'],
        'jpeg': [b'\xFF\xD8\xFF'],
        'png': [b'\x89PNG\r\n\x1a\n'],
        'webp': [b'RIFF'],
        'mp3': [b'ID3', b'\xFF\xFB'],
        'wav': [b'RIFF'],
        'm4a': [b'\x00\x00\x00\x20ftypM4A'],
        'ogg': [b'OggS']
    }
    
    if extension in signatures:
        for sig in signatures[extension]:
            if header.startswith(sig):
                return True
        return False
    
    # Default to true for unknown extensions
    return True

# === HASHING AND ENCODING UTILITIES ===

def generate_claim_hash(claim: str) -> str:
    """Generate unique hash for claim caching and deduplication"""
    if not claim:
        return ""
    
    # Normalize text for consistent hashing
    normalized = normalize_text(claim.lower())
    
    # Remove punctuation for better deduplication
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    timestamp = str(int(time.time() * 1000))  # milliseconds
    random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"req_{timestamp}_{random_part}"

def encode_base64_safe(data: bytes) -> str:
    """Safely encode bytes to base64 string"""
    try:
        return base64.b64encode(data).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 encoding failed: {e}")
        return ""

def decode_base64_safe(data: str) -> Optional[bytes]:
    """Safely decode base64 string to bytes"""
    try:
        return base64.b64decode(data)
    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        return None

# === TIME AND DATE UTILITIES ===

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + 'Z'

def get_timestamp_ago(hours: int) -> str:
    """Get timestamp from hours ago"""
    past_time = datetime.utcnow() - timedelta(hours=hours)
    return past_time.isoformat() + 'Z'

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# === CULTURAL CONTEXT ANALYSIS ===

def analyze_cultural_context(text: str) -> Dict[str, Any]:
    """Analyze text for Indian cultural context and markers"""
    if not text:
        return {}
    
    text_lower = text.lower()
    context = {
        'festivals_mentioned': [],
        'government_schemes': [],
        'political_references': [],
        'regional_markers': [],
        'health_terms': [],
        'cultural_sensitivity': 'low',
        'regional_specific': False,
        'language_mixing': False
    }
    
    # Check for cultural patterns
    for category, patterns in CULTURAL_PATTERNS.items():
        if category in context:
            for pattern in patterns:
                if pattern in text_lower:
                    context[category].append(pattern)
    
    # Check for language mixing (Hinglish detection)
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    
    if english_words > 0 and devanagari_chars > 0:
        context['language_mixing'] = True
    
    # Determine regional specificity
    context['regional_specific'] = bool(context['regional_markers'])
    
    # Calculate sensitivity level
    sensitivity_score = 0
    if context['political_references']:
        sensitivity_score += 3
    if context['festivals_mentioned']:
        sensitivity_score += 1
    if context['government_schemes']:
        sensitivity_score += 2
    
    if sensitivity_score >= 5:
        context['cultural_sensitivity'] = 'high'
    elif sensitivity_score >= 2:
        context['cultural_sensitivity'] = 'medium'
    
    return context

def detect_manipulation_patterns(text: str) -> Dict[str, Any]:
    """Detect psychological manipulation patterns in text"""
    if not text:
        return {}
    
    text_lower = text.lower()
    patterns = {
        'emotional_triggers': [],
        'urgency_tactics': [],
        'authority_claims': [],
        'social_proof': [],
        'fear_appeals': [],
        'manipulation_score': 0
    }
    
    # Check for each manipulation pattern
    for pattern_type, indicators in MANIPULATION_INDICATORS.items():
        for indicator in indicators:
            if indicator in text_lower:
                patterns[pattern_type].append(indicator)
    
    # Calculate manipulation score
    score = 0
    score += len(patterns['emotional_triggers']) * 15
    score += len(patterns['urgency_tactics']) * 20
    score += len(patterns['authority_claims']) * 25
    score += len(patterns['social_proof']) * 10
    score += len(patterns['fear_appeals']) * 30
    
    patterns['manipulation_score'] = min(score, 100)
    
    return patterns

# === CONTENT SAFETY UTILITIES ===

def check_content_safety(text: str) -> Dict[str, Any]:
    """Check content for harmful patterns"""
    if not text:
        return {'is_safe': True, 'risk_level': 'none', 'flags': []}
    
    text_lower = text.lower()
    safety_check = {
        'is_safe': True,
        'risk_level': 'none',
        'flags': [],
        'reasons': []
    }
    
    for category, patterns in HARMFUL_CONTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                safety_check['flags'].append(category)
                safety_check['reasons'].append(f"Contains {category}: {pattern}")
                safety_check['is_safe'] = False
    
    # Determine risk level
    if len(safety_check['flags']) >= 3:
        safety_check['risk_level'] = 'high'
    elif len(safety_check['flags']) >= 1:
        safety_check['risk_level'] = 'medium'
    elif any(flag in ['violence', 'self_harm'] for flag in safety_check['flags']):
        safety_check['risk_level'] = 'high'
    
    return safety_check

# === FILE TYPE UTILITIES ===

def get_file_mime_type(filename: str) -> str:
    """Get MIME type of file based on extension"""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'

def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension"""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    return extension in VALIDATION_RULES['allowed_image_formats']

def is_audio_file(filename: str) -> bool:
    """Check if file is an audio file based on extension"""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    return extension in VALIDATION_RULES['allowed_audio_formats']

# === JSON UTILITIES ===

def safe_json_loads(data: str, default=None) -> Any:
    """Safely parse JSON with fallback"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default=None) -> str:
    """Safely serialize to JSON with fallback"""
    try:
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        if default is not None:
            return json.dumps(default)
        return "{}"

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)

# === ERROR HANDLING UTILITIES ===

def create_error_response(error_code: str, message: str = None, details: Dict = None) -> Dict:
    """Create standardized error response"""
    response = {
        'success': False,
        'error': {
            'code': error_code,
            'message': message or ERROR_MESSAGES.get(error_code, 'Unknown error'),
            'timestamp': get_current_timestamp()
        }
    }
    
    if details:
        response['error']['details'] = sanitize_for_json(details)
    
    return response

def create_success_response(data: Any, message: str = None) -> Dict:
    """Create standardized success response"""
    return {
        'success': True,
        'data': sanitize_for_json(data),
        'message': message,
        'timestamp': get_current_timestamp()
    }

# === PERFORMANCE MONITORING ===

class PerformanceTimer:
    """Context manager for measuring execution time"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.operation_name} completed in {format_duration(duration)}")
        
        if exc_type:
            logger.error(f"{self.operation_name} failed: {exc_val}")

# === SIMPLE IN-MEMORY CACHE ===

class SimpleCache:
    """Simple in-memory cache for development"""
    
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._access_times = {}
        self.max_size = max_size
    
    def get(self, key: str, default=None):
        """Get value from cache"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return default
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if len(self._cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def delete(self, key: str):
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
    
    def clear(self):
        """Clear all cache"""
        self._cache.clear()
        self._access_times.clear()

# Global cache instance
cache = SimpleCache()

def extract_temporal_indicators(text: str) -> Dict[str, Any]:
    """Extract temporal context and current event indicators"""
    from datetime import datetime
    
    current_year = datetime.now().year
    text_lower = text.lower()
    
    # Extract years mentioned
    year_matches = re.findall(r'\b(202[0-9])\b', text)
    
    # Recent/current event indicators
    recent_indicators = [
        'latest', 'recent', 'today', 'yesterday', 'breaking', 'just announced',
        'currently', 'now', 'this year', 'this month', 'recently'
    ]
    
    # Future event indicators
    future_indicators = ['will', 'going to', 'scheduled', 'upcoming', 'next']
    
    context = {
        'years_mentioned': year_matches,
        'has_recent_indicators': any(indicator in text_lower for indicator in recent_indicators),
        'has_future_indicators': any(indicator in text_lower for indicator in future_indicators),
        'mentions_current_year': str(current_year) in text,
        'mentions_future_year': any(int(year) > current_year for year in year_matches if year.isdigit()),
        'temporal_sensitivity': 'high' if any(indicator in text_lower for indicator in recent_indicators + future_indicators) else 'low'
    }
    
    return context

# === TRANSLATION UTILITIES ===

def translate_text(text: str, target_language: str = 'en', source_language: str = 'auto') -> Tuple[str, str]:
    """Translate text using Google Translate API with caching and fallback"""
    if not text or not text.strip():
        return text, source_language

    # Skip translation if already in target language
    if source_language == target_language:
        return text, source_language

    try:
        from config import GOOGLE_SEARCH_API_KEY, GOOGLE_TRANSLATE_ENDPOINT, LANGUAGE_MAPPINGS

        if not GOOGLE_SEARCH_API_KEY:
            logger.warning("Translation API key not available, translation skipped")
            return text, source_language
        
        # Check cache first
        cache_key = f"translate:{hashlib.md5(f'{text}{source_language}{target_language}'.encode()).hexdigest()}"
        cached_translation = cache.get(cache_key)
        if cached_translation:
            return cached_translation['text'], cached_translation['detected_language']
        
        # Map language codes for translation API
        target_code = LANGUAGE_MAPPINGS.get(target_language, {}).get('translate', target_language)
        source_code = LANGUAGE_MAPPINGS.get(source_language, {}).get('translate', source_language) if source_language != 'auto' else 'auto'
        
        # Call Google Translate API
        url = f"{GOOGLE_TRANSLATE_ENDPOINT}?key={GOOGLE_SEARCH_API_KEY}"
        data = {
            'q': text,
            'target': target_code,
            'format': 'text'
        }
        
        if source_code != 'auto':
            data['source'] = source_code
        
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if 'data' in result and 'translations' in result['data']:
            translation = result['data']['translations'][0]
            translated_text = translation['translatedText']
            detected_lang = translation.get('detectedSourceLanguage', source_language)
            
            # Cache the result
            cache.set(cache_key, {
                'text': translated_text,
                'detected_language': detected_lang
            }, ttl=3600)
            
            logger.info(f"Translated text from {detected_lang} to {target_language}")
            return translated_text, detected_lang
        
        logger.warning("Translation API returned unexpected format")
        return text, source_language
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text, source_language

def get_language_service_code(language: str, service: str) -> str:
    """Get the correct language code for a specific service"""
    from config import LANGUAGE_MAPPINGS
    return LANGUAGE_MAPPINGS.get(language, {}).get(service, language)

def detect_and_translate_if_needed(text: str, target_language: str = 'en') -> Dict[str, str]:
    """Detect language and translate if needed"""
    detected_language = detect_language_heuristic(text)
    
    if detected_language == target_language:
        return {
            'original_text': text,
            'translated_text': text,
            'original_language': detected_language,
            'target_language': target_language,
            'was_translated': False
        }
    
    translated_text, confirmed_language = translate_text(text, target_language, detected_language)
    
    return {
        'original_text': text,
        'translated_text': translated_text,
        'original_language': confirmed_language,
        'target_language': target_language,
        'was_translated': True
    }

logger.info("Utils module initialized successfully")