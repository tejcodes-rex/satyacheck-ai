"""
SatyaCheck AI - Enhanced Configuration
Optimized configuration with better validation for Google Gen AI Exchange Hackathon
"""

import os
import logging
from typing import Dict, List, Optional
from enum import Enum

# === ENVIRONMENT CONFIGURATION ===

class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

ENVIRONMENT = Environment(os.getenv('ENVIRONMENT', 'production'))

# === GOOGLE CLOUD CONFIGURATION ===

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'satyacheck-ai')
REGION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')

# === API CONFIGURATION ===

# Import secret_manager module for secure credential management
# FIXED: Renamed from 'secrets' to 'secret_manager' to avoid conflict with Python's built-in secrets module
try:
    from secret_manager import get_gemini_api_key, get_google_search_api_key, get_google_search_engine_id
    _SECRETS_AVAILABLE = True
except ImportError:
    _SECRETS_AVAILABLE = False
    logging.warning("Secret manager module not available, falling back to environment variables")

# Required API Keys - Retrieved securely from Secret Manager with fallback
if _SECRETS_AVAILABLE:
    GEMINI_API_KEY = get_gemini_api_key()
    GOOGLE_SEARCH_API_KEY = get_google_search_api_key()
    GOOGLE_SEARCH_ENGINE_ID = get_google_search_engine_id()
else:
    # Fallback to environment variables for local development
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY')
    GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

# API Endpoints
# Using gemini-2.5-flash (stable model available in the API)
GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GOOGLE_TRANSLATE_PARENT = f"projects/{PROJECT_ID}/locations/global"
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# === TRANSLATION CONFIGURATION ===
GOOGLE_TRANSLATE_ENDPOINT = f"https://translation.googleapis.com/language/translate/v2"
TRANSLATION_CACHE_TTL = 3600  # 1 hour cache for translations

# Language code mappings for different services
LANGUAGE_MAPPINGS = {
    'hi': {'gemini': 'hi', 'translate': 'hi', 'speech': 'hi-IN', 'tts': 'hi-IN'},
    'en': {'gemini': 'en', 'translate': 'en', 'speech': 'en-IN', 'tts': 'en-IN'},
    'ta': {'gemini': 'ta', 'translate': 'ta', 'speech': 'ta-IN', 'tts': 'ta-IN'},
    'te': {'gemini': 'te', 'translate': 'te', 'speech': 'te-IN', 'tts': 'te-IN'},
    'bn': {'gemini': 'bn', 'translate': 'bn', 'speech': 'bn-IN', 'tts': 'bn-IN'},
    'mr': {'gemini': 'mr', 'translate': 'mr', 'speech': 'mr-IN', 'tts': 'mr-IN'},
    'gu': {'gemini': 'gu', 'translate': 'gu', 'speech': 'gu-IN', 'tts': 'gu-IN'}
}

# === LANGUAGE CONFIGURATION ===

SUPPORTED_LANGUAGES = {
    'hi': {'name': 'Hindi', 'native': 'हिंदी', 'tts_code': 'hi-IN'},
    'en': {'name': 'English', 'native': 'English', 'tts_code': 'en-IN'},
    'ta': {'name': 'Tamil', 'native': 'தமிழ்', 'tts_code': 'ta-IN'},
    'te': {'name': 'Telugu', 'native': 'తెలుగు', 'tts_code': 'te-IN'},
    'bn': {'name': 'Bengali', 'native': 'বাংলা', 'tts_code': 'bn-IN'},
    'mr': {'name': 'Marathi', 'native': 'मराठी', 'tts_code': 'mr-IN'},
    'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી', 'tts_code': 'gu-IN'}
}

DEFAULT_LANGUAGE = 'en'

# === CULTURAL CONTEXT PATTERNS ===

CULTURAL_PATTERNS = {
    'festivals': [
        'diwali', 'deepavali', 'holi', 'eid', 'christmas', 'dussehra', 'vijaya dashami',
        'navratri', 'ganesh chaturthi', 'karva chauth', 'raksha bandhan', 'rakhi',
        'janmashtami', 'durga puja', 'kali puja', 'onam', 'pongal', 'makar sankranti',
        'baisakhi', 'gudi padwa', 'ugadi', 'poila boishakh', 'vishu', 'bihu'
    ],
    'government_schemes': [
        'pm kisan', 'pradhan mantri kisan', 'ayushman bharat', 'jan aushadhi',
        'pradhan mantri', 'swachh bharat', 'digital india', 'make in india',
        'skill india', 'startup india', 'beti bachao', 'ujjwala yojana',
        'jal jeevan mission', 'atmanirbhar bharat', 'mudra loan', 'jan dhan'
    ],
    'political_terms': [
        'bjp', 'congress', 'aap', 'aam aadmi party', 'election', 'parliament',
        'lok sabha', 'rajya sabha', 'chief minister', 'cm', 'prime minister', 'pm',
        'modi', 'rahul gandhi', 'kejriwal', 'election commission', 'evm'
    ],
    'health_terms': [
        'ayurveda', 'homeopathy', 'covid', 'coronavirus', 'vaccine', 'vaccination',
        'covishield', 'covaxin', 'immunity', 'kadha', 'tulsi', 'turmeric', 'haldi'
    ],
    'regional_markers': [
        'punjab', 'haryana', 'rajasthan', 'gujarat', 'maharashtra', 'karnataka',
        'kerala', 'tamil nadu', 'andhra pradesh', 'telangana', 'west bengal',
        'bihar', 'uttar pradesh', 'delhi', 'mumbai', 'kolkata', 'chennai'
    ]
}

# === SOURCE CREDIBILITY CONFIGURATION ===

SOURCE_CREDIBILITY_SCORES = {
    # Tier 1: Highest credibility (0.90-0.95) - Government and International
    'pib.gov.in': 0.95,
    'reuters.com': 0.92,
    'bbc.com': 0.90,
    'ap.org': 0.91,
    
    # Tier 2: High credibility (0.80-0.89) - Established fact-checkers and quality news
    'factchecker.in': 0.88,
    'boomlive.in': 0.85,
    'altnews.in': 0.85,
    'thehindu.com': 0.83,
    'indianexpress.com': 0.82,
    'hindustantimes.com': 0.80,
    
    # Tier 3: Moderate credibility (0.70-0.79) - Mainstream media
    'ndtv.com': 0.75,
    'cnn.com': 0.72,
    'wikipedia.org': 0.70,
    'timesofindia.indiatimes.com': 0.68,
    
    # Tier 4: Lower credibility (0.40-0.60) - Social media and aggregators
    'twitter.com': 0.45,
    'facebook.com': 0.40,
    'whatsapp': 0.20,
    'youtube.com': 0.50,
    
    # Default for unknown sources
    'unknown': 0.50
}

DOMAIN_EXPERTISE_BONUS = {
    'health': ['who.int', 'mohfw.gov.in', 'cdc.gov', 'icmr.gov.in'],
    'politics': ['eci.gov.in', 'parliamentofindia.nic.in'],
    'sports': ['bcci.tv', 'icc-cricket.com', 'fifa.com'],
    'technology': ['cert-in.org.in', 'meity.gov.in']
}

# === CATEGORY EXPERT SOURCES ===
# Authoritative sources organized by claim category for enhanced credibility

CATEGORY_EXPERT_SOURCES = {
    'economy': [
        {
            'title': 'International Monetary Fund - World Economic Outlook',
            'url': 'https://www.imf.org/en/Publications/WEO',
            'publisher': 'International Monetary Fund',
            'description': 'Official IMF economic data and GDP rankings',
            'credibilityScore': 10
        },
        {
            'title': 'World Bank - World Development Indicators',
            'url': 'https://datatopics.worldbank.org/world-development-indicators/',
            'publisher': 'World Bank',
            'description': 'Comprehensive global economic statistics',
            'credibilityScore': 10
        },
        {
            'title': 'Reserve Bank of India',
            'url': 'https://www.rbi.org.in/',
            'publisher': 'Reserve Bank of India',
            'description': 'Official Indian economic and monetary data',
            'credibilityScore': 10
        },
        {
            'title': 'Ministry of Finance - India',
            'url': 'https://www.finmin.nic.in/',
            'publisher': 'Ministry of Finance, Government of India',
            'description': 'Official government financial information',
            'credibilityScore': 10
        }
    ],
    'health': [
        {
            'title': 'World Health Organization',
            'url': 'https://www.who.int/',
            'publisher': 'WHO',
            'description': 'Global health information and disease tracking',
            'credibilityScore': 10
        },
        {
            'title': 'Ministry of Health and Family Welfare',
            'url': 'https://www.mohfw.gov.in/',
            'publisher': 'MOHFW, Government of India',
            'description': 'Official Indian health policies and updates',
            'credibilityScore': 10
        },
        {
            'title': 'Indian Council of Medical Research',
            'url': 'https://www.icmr.gov.in/',
            'publisher': 'ICMR',
            'description': 'Leading medical research institution in India',
            'credibilityScore': 10
        },
        {
            'title': 'Centers for Disease Control and Prevention',
            'url': 'https://www.cdc.gov/',
            'publisher': 'CDC',
            'description': 'US national public health institute',
            'credibilityScore': 9
        }
    ],
    'politics': [
        {
            'title': 'Election Commission of India',
            'url': 'https://eci.gov.in/',
            'publisher': 'Election Commission of India',
            'description': 'Official election results and information',
            'credibilityScore': 10
        },
        {
            'title': 'Parliament of India',
            'url': 'https://sansad.in/',
            'publisher': 'Parliament of India',
            'description': 'Official parliamentary proceedings and bills',
            'credibilityScore': 10
        },
        {
            'title': 'Press Information Bureau',
            'url': 'https://pib.gov.in/',
            'publisher': 'PIB, Government of India',
            'description': 'Official government press releases',
            'credibilityScore': 10
        },
        {
            'title': 'MyNeta - Candidate Affidavits',
            'url': 'https://myneta.info/',
            'publisher': 'Association for Democratic Reforms',
            'description': 'Verified candidate background information',
            'credibilityScore': 8
        }
    ],
    'sports': [
        {
            'title': 'International Cricket Council',
            'url': 'https://www.icc-cricket.com/',
            'publisher': 'ICC',
            'description': 'Official international cricket body',
            'credibilityScore': 10
        },
        {
            'title': 'Board of Control for Cricket in India',
            'url': 'https://www.bcci.tv/',
            'publisher': 'BCCI',
            'description': 'Official Indian cricket board',
            'credibilityScore': 10
        },
        {
            'title': 'Olympics Official Website',
            'url': 'https://olympics.com/',
            'publisher': 'International Olympic Committee',
            'description': 'Official Olympic Games information',
            'credibilityScore': 10
        },
        {
            'title': 'FIFA',
            'url': 'https://www.fifa.com/',
            'publisher': 'FIFA',
            'description': 'Official international football body',
            'credibilityScore': 10
        }
    ],
    'science': [
        {
            'title': 'Indian Space Research Organisation',
            'url': 'https://www.isro.gov.in/',
            'publisher': 'ISRO',
            'description': 'Official Indian space agency',
            'credibilityScore': 10
        },
        {
            'title': 'Council of Scientific & Industrial Research',
            'url': 'https://www.csir.res.in/',
            'publisher': 'CSIR',
            'description': 'Premier Indian R&D organization',
            'credibilityScore': 10
        },
        {
            'title': 'NASA',
            'url': 'https://www.nasa.gov/',
            'publisher': 'NASA',
            'description': 'US space agency and research',
            'credibilityScore': 10
        },
        {
            'title': 'Nature Journal',
            'url': 'https://www.nature.com/',
            'publisher': 'Nature Publishing Group',
            'description': 'Leading peer-reviewed science journal',
            'credibilityScore': 9
        }
    ],
    'technology': [
        {
            'title': 'CERT-In',
            'url': 'https://www.cert-in.org.in/',
            'publisher': 'Indian Computer Emergency Response Team',
            'description': 'Official cybersecurity alerts and advisories',
            'credibilityScore': 10
        },
        {
            'title': 'Ministry of Electronics & IT',
            'url': 'https://www.meity.gov.in/',
            'publisher': 'MeitY, Government of India',
            'description': 'Official IT policies and regulations',
            'credibilityScore': 10
        },
        {
            'title': 'IEEE Spectrum',
            'url': 'https://spectrum.ieee.org/',
            'publisher': 'IEEE',
            'description': 'Leading technology publication',
            'credibilityScore': 9
        }
    ],
    'environment': [
        {
            'title': 'Ministry of Environment, Forest and Climate Change',
            'url': 'https://moef.gov.in/',
            'publisher': 'MoEFCC, Government of India',
            'description': 'Official environmental policies',
            'credibilityScore': 10
        },
        {
            'title': 'India Meteorological Department',
            'url': 'https://mausam.imd.gov.in/',
            'publisher': 'IMD',
            'description': 'Official weather and climate data',
            'credibilityScore': 10
        },
        {
            'title': 'IPCC - Climate Change Reports',
            'url': 'https://www.ipcc.ch/',
            'publisher': 'Intergovernmental Panel on Climate Change',
            'description': 'UN climate science assessments',
            'credibilityScore': 10
        }
    ],
    'education': [
        {
            'title': 'Ministry of Education',
            'url': 'https://www.education.gov.in/',
            'publisher': 'Ministry of Education, Government of India',
            'description': 'Official education policies and schemes',
            'credibilityScore': 10
        },
        {
            'title': 'University Grants Commission',
            'url': 'https://www.ugc.gov.in/',
            'publisher': 'UGC',
            'description': 'Higher education regulation and standards',
            'credibilityScore': 10
        }
    ]
}

# === COMPREHENSIVE FACT-CHECKING SOURCES ===
# Regional and international fact-checking platforms

FACT_CHECKING_SOURCES = {
    'indian': [
        {
            'title': 'PIB Fact Check',
            'url': 'https://pib.gov.in/factcheck.aspx',
            'publisher': 'Press Information Bureau',
            'description': 'Official government fact-checking portal',
            'credibilityScore': 10
        },
        {
            'title': 'FactChecker.in',
            'url': 'https://www.factchecker.in/',
            'publisher': 'FactChecker.in',
            'description': 'India\'s first data-driven fact-checking initiative',
            'credibilityScore': 9
        },
        {
            'title': 'BOOM Live',
            'url': 'https://www.boomlive.in/',
            'publisher': 'BOOM',
            'description': 'Independent fact-checking platform',
            'credibilityScore': 9
        },
        {
            'title': 'Alt News',
            'url': 'https://www.altnews.in/',
            'publisher': 'Alt News',
            'description': 'Fact-checking and debunking platform',
            'credibilityScore': 9
        },
        {
            'title': 'Vishvas News',
            'url': 'https://www.vishvasnews.com/',
            'publisher': 'Vishvas News',
            'description': 'Fact-checking in multiple Indian languages',
            'credibilityScore': 8
        },
        {
            'title': 'Newsmobile Fact Check',
            'url': 'https://newsmobile.in/section/fact-check/',
            'publisher': 'Newsmobile',
            'description': 'Mobile-first fact-checking',
            'credibilityScore': 8
        }
    ],
    'international': [
        {
            'title': 'Snopes',
            'url': 'https://www.snopes.com/',
            'publisher': 'Snopes',
            'description': 'Oldest and largest fact-checking site',
            'credibilityScore': 9
        },
        {
            'title': 'Reuters Fact Check',
            'url': 'https://www.reuters.com/fact-check',
            'publisher': 'Reuters',
            'description': 'Global news agency fact-checking',
            'credibilityScore': 9
        },
        {
            'title': 'AFP Fact Check',
            'url': 'https://factcheck.afp.com/',
            'publisher': 'AFP',
            'description': 'International fact-checking network',
            'credibilityScore': 9
        },
        {
            'title': 'Full Fact',
            'url': 'https://fullfact.org/',
            'publisher': 'Full Fact',
            'description': 'UK\'s independent fact-checking charity',
            'credibilityScore': 9
        }
    ]
}
# === MANIPULATION INDICATORS ===

MANIPULATION_INDICATORS = {
    'emotional_triggers': [
        'shocking', 'unbelievable', 'amazing', 'terrible', 'devastating',
        'heartbreaking', 'incredible', 'outrageous', 'horrific', 'miraculous'
    ],
    'urgency_tactics': [
        'urgent', 'immediate', 'quickly', 'before it\'s too late', 'deadline',
        'limited time', 'act now', 'don\'t wait', 'hurry', 'breaking'
    ],
    'authority_claims': [
        'doctor says', 'expert confirms', 'scientist proves', 'government announces',
        'study shows', 'research reveals', 'official statement', 'according to sources'
    ],
    'social_proof': [
        'everyone knows', 'millions of people', 'viral video', 'trending now',
        'shared by thousands', 'going viral', 'most people believe'
    ],
    'fear_appeals': [
        'danger', 'threat', 'risk', 'deadly', 'harmful', 'toxic', 'poisonous',
        'kill', 'death', 'disease', 'epidemic', 'crisis', 'emergency'
    ]
}

# === VALIDATION RULES ===

VALIDATION_RULES = {
    'max_claim_length': 5000,
    'min_claim_length': 10,
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_image_formats': ['jpg', 'jpeg', 'png', 'webp'],
    'allowed_audio_formats': ['mp3', 'wav', 'ogg', 'm4a', 'webm'],
    'max_url_length': 2048
}

# === ERROR MESSAGES ===

ERROR_MESSAGES = {
    'invalid_input': 'Invalid input provided. Please check your data.',
    'file_too_large': f'File too large. Maximum size is {VALIDATION_RULES["max_file_size"] // (1024*1024)}MB.',
    'unsupported_format': 'Unsupported file format.',
    'processing_failed': 'Processing failed. Please try again.',
    'api_unavailable': 'Service temporarily unavailable. Please try again later.',
    'claim_too_long': f'Claim too long. Maximum length is {VALIDATION_RULES["max_claim_length"]} characters.',
    'claim_too_short': f'Claim too short. Minimum length is {VALIDATION_RULES["min_claim_length"]} characters.'
}

# === SUCCESS MESSAGES ===

SUCCESS_MESSAGES = {
    'analysis_complete': 'Analysis completed successfully.',
    'health_check_passed': 'All systems operational.'
}

# === CONTENT SAFETY PATTERNS ===

HARMFUL_CONTENT_PATTERNS = {
    'violence': [
        'kill', 'murder', 'attack', 'bomb', 'terrorist', 'violence',
        'assault', 'lynch', 'mob violence'
    ],
    'hate_speech': [
        'hate', 'discriminat', 'racist', 'communal violence',
        'religious hatred', 'casteist', 'anti-national'
    ],
    'self_harm': [
        'suicide', 'self harm', 'end life', 'kill myself'
    ]
}

# === LOGGING CONFIGURATION ===

LOG_CONFIG = {
    'level': 'INFO' if ENVIRONMENT == Environment.PRODUCTION else 'DEBUG',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# === UTILITY FUNCTIONS ===

def is_production() -> bool:
    """Check if running in production"""
    return ENVIRONMENT == Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development"""
    return ENVIRONMENT == Environment.DEVELOPMENT

def validate_language(language_code: str) -> bool:
    """Validate if language code is supported"""
    return language_code in SUPPORTED_LANGUAGES

def get_config_value(key: str, default=None):
    """Get configuration value with fallback"""
    return globals().get(key, default)

# === ENHANCED ENVIRONMENT VALIDATION ===

def validate_environment() -> List[str]:
    """Validate required environment variables and return issues"""
    issues = []
    
    # Critical requirements
    if not GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY not set - AI analysis will fail")
    
    # Check if PROJECT_ID is properly configured
    if not PROJECT_ID or PROJECT_ID == 'satyacheck-ai':
        issues.append("GOOGLE_CLOUD_PROJECT not properly configured - using default")
    
    # Optional but recommended for full functionality
    if not GOOGLE_SEARCH_API_KEY:
        issues.append("GOOGLE_SEARCH_API_KEY not set - PIB search will be disabled")
    
    if not GOOGLE_SEARCH_ENGINE_ID:
        issues.append("GOOGLE_SEARCH_ENGINE_ID not set - PIB search will be disabled")
    
    # Validate Gemini API key format (basic check)
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('AIza'):
        issues.append("GEMINI_API_KEY format appears invalid")
    
    return issues

def get_deployment_config() -> Dict[str, any]:
    """Get deployment configuration summary"""
    config_issues = validate_environment()
    
    return {
        'environment': ENVIRONMENT.value,
        'project_id': PROJECT_ID,
        'region': REGION,
        'gemini_configured': bool(GEMINI_API_KEY),
        'search_configured': bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID),
        'supported_languages': list(SUPPORTED_LANGUAGES.keys()),
        'issues': config_issues,
        'ready_for_deployment': len([issue for issue in config_issues if 'will fail' in issue]) == 0
    }

def log_startup_config():
    """Log startup configuration for debugging"""
    logger = logging.getLogger(__name__)
    
    config = get_deployment_config()
    
    logger.info("=== SatyaCheck AI Configuration ===")
    logger.info(f"Environment: {config['environment']}")
    logger.info(f"Project ID: {config['project_id']}")
    logger.info(f"Region: {config['region']}")
    logger.info(f"Gemini API: {'✓ Configured' if config['gemini_configured'] else '✗ Missing'}")
    logger.info(f"Search API: {'✓ Configured' if config['search_configured'] else '✗ Missing'}")
    logger.info(f"Languages: {', '.join(config['supported_languages'])}")
    
    if config['issues']:
        logger.warning("Configuration Issues:")
        for issue in config['issues']:
            logger.warning(f"  - {issue}")
    
    logger.info(f"Deployment Ready: {'✓ Yes' if config['ready_for_deployment'] else '✗ No'}")
    logger.info("=" * 40)

# === SECURITY CONFIGURATION ===

SECURITY_CONFIG = {
    'max_requests_per_minute': 60,
    'max_file_uploads_per_hour': 100,
    'allowed_origins': ['*'],  # Allow all origins for testing and production
    'rate_limit_enabled': is_production(),
    'content_length_limit': 10 * 1024 * 1024,  # 10MB for request body
    'timeout_seconds': 30
}

# === PERFORMANCE CONFIGURATION ===

PERFORMANCE_CONFIG = {
    'cache_ttl_seconds': 3600,  # 1 hour
    'max_cache_size': 1000,
    'connection_pool_size': 10,
    'request_timeout': 30,
    'retry_attempts': 3,
    'backoff_factor': 1.5
}

# === FEATURE FLAGS ===

FEATURE_FLAGS = {
    'enable_pib_search': bool(GOOGLE_SEARCH_API_KEY),
    'enable_audio_processing': True,
    'enable_live_audio': True,
    'enable_image_processing': True,
    'enable_url_processing': True,
    'enable_analytics': True,
    'enable_caching': True,
    'enable_cultural_analysis': True,
    'enable_manipulation_detection': True,
    'enable_viral_potential_analysis': True
}

# Initialize logging and validation on import
if __name__ != '__main__':
    try:
        logging.basicConfig(
            level=getattr(logging, LOG_CONFIG['level']),
            format=LOG_CONFIG['format']
        )
        
        # Log startup configuration in development
        if is_development():
            log_startup_config()
        
        # Validate critical configuration
        issues = validate_environment()
        critical_issues = [issue for issue in issues if 'will fail' in issue]
        
        if critical_issues and is_production():
            raise ValueError(f"Critical configuration issues: {'; '.join(critical_issues)}")
            
    except Exception as e:
        print(f"Configuration initialization failed: {e}")
        if is_production():
            raise

# Export configuration summary
__all__ = [
    'ENVIRONMENT', 'PROJECT_ID', 'REGION',
    'GEMINI_API_KEY', 'GOOGLE_SEARCH_API_KEY', 'GOOGLE_SEARCH_ENGINE_ID',
    'GEMINI_API_ENDPOINT', 'GOOGLE_TRANSLATE_PARENT', 'GOOGLE_SEARCH_URL',
    'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE',
    'CULTURAL_PATTERNS', 'MANIPULATION_INDICATORS',
    'VALIDATION_RULES', 'ERROR_MESSAGES', 'SUCCESS_MESSAGES',
    'HARMFUL_CONTENT_PATTERNS', 'LOG_CONFIG',
    'SECURITY_CONFIG', 'PERFORMANCE_CONFIG', 'FEATURE_FLAGS',
    'is_production', 'is_development', 'validate_language',
    'validate_environment', 'get_deployment_config',
    'SOURCE_CREDIBILITY_SCORES', 'DOMAIN_EXPERTISE_BONUS',
    'CATEGORY_EXPERT_SOURCES', 'FACT_CHECKING_SOURCES',
    'GOOGLE_TRANSLATE_ENDPOINT', 'TRANSLATION_CACHE_TTL', 'LANGUAGE_MAPPINGS'
]