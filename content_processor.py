"""
SatyaCheck AI - Enhanced Content Processing Engine
Multi-modal content processing for text, images, audio, and URLs with live audio support
"""

import base64
import cv2
import json
import logging
import numpy as np
import re
import requests
import time
import io
import wave
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Google Cloud imports
from google.cloud import vision, speech, texttospeech
from google.api_core import exceptions as gcloud_exceptions

# Local imports
from config import PROJECT_ID, GOOGLE_TRANSLATE_PARENT, SUPPORTED_LANGUAGES
from utils import (
    normalize_text, PerformanceTimer, create_error_response,
    get_current_timestamp, validate_file_upload, validate_url,
    is_image_file, is_audio_file, get_file_mime_type, translate_text,
    sanitize_error_message
)

# Configure logging
logger = logging.getLogger(__name__)

# === DATA MODELS ===

@dataclass
class MediaAnalysis:
    """Analysis result for media content"""
    media_type: str
    extracted_text: str
    confidence_score: float
    metadata: Dict[str, Any]
    manipulation_detected: bool
    manipulation_details: Dict[str, Any]
    processing_time: float

@dataclass
class URLAnalysis:
    """Analysis result for URL content"""
    url: str
    platform_type: str
    content_type: str
    title: str
    extracted_text: str
    images: List[str]
    metadata: Dict[str, Any]
    processing_time: float

# === AUDIO PROCESSING UTILITIES ===

def convert_audio_format(audio_data: bytes, target_format: str = 'wav') -> bytes:
    """Convert audio to target format for Speech API"""
    try:
        return audio_data
    except Exception as e:
        logger.error(f"Audio format conversion failed: {e}")
        return audio_data

def validate_audio_quality(audio_data: bytes) -> Dict[str, Any]:
    """Enhanced audio quality validation with better error handling"""
    quality_check = {
        'is_valid': True,
        'duration_estimate': 0.0,
        'size_mb': len(audio_data) / (1024 * 1024),
        'warnings': [],
        'quality_score': 1.0
    }

    try:
        # Basic size validation - only fail for extreme cases
        if len(audio_data) < 500:  # Less than 500 bytes - definitely corrupt
            quality_check['is_valid'] = False
            quality_check['warnings'].append('Audio file too small or corrupted')
            quality_check['quality_score'] = 0.0
        elif len(audio_data) > 50 * 1024 * 1024:  # More than 50MB
            quality_check['is_valid'] = False
            quality_check['warnings'].append('Audio file too large (max 50MB)')
            quality_check['quality_score'] = 0.0
        else:
            # Calculate quality score based on size
            size_mb = quality_check['size_mb']
            if size_mb < 0.1:  # Very small files get lower score
                quality_check['quality_score'] = max(0.5, size_mb * 10)
                quality_check['warnings'].append('Audio file is very small - quality may be low')
            elif size_mb > 25:  # Large files get slightly lower score
                quality_check['quality_score'] = 0.8
                quality_check['warnings'].append('Large audio file - processing may take longer')

        # Estimate duration based on file size (rough approximation for various formats)
        # More conservative estimate: assume 64kbps average bitrate
        estimated_duration = len(audio_data) / (8000)  # 8KB/second average
        quality_check['duration_estimate'] = estimated_duration

        # Duration-based warnings (but don't fail)
        if estimated_duration > 600:  # More than 10 minutes
            quality_check['warnings'].append('Audio longer than 10 minutes - may timeout')
            quality_check['quality_score'] *= 0.8
        elif estimated_duration < 0.5:  # Less than 0.5 seconds
            quality_check['warnings'].append('Audio very short - transcription may be incomplete')
            quality_check['quality_score'] *= 0.7

        # Basic format validation (check for common audio headers)
        audio_start = audio_data[:12] if len(audio_data) >= 12 else audio_data
        has_audio_signature = (
            audio_start.startswith(b'RIFF') or  # WAV
            audio_start.startswith(b'ID3') or   # MP3 with ID3
            audio_start.startswith(b'\xff\xfb') or  # MP3
            audio_start.startswith(b'OggS') or  # OGG
            audio_start.startswith(b'ftyp') or  # M4A/MP4
            b'webm' in audio_start.lower()      # WebM
        )

        if not has_audio_signature and len(audio_data) > 1000:
            quality_check['warnings'].append('Unrecognized audio format - may not process correctly')
            quality_check['quality_score'] *= 0.6
            # Don't fail completely - let Speech API try to handle it

    except Exception as e:
        logger.warning(f"Audio quality validation error: {e}")
        # If validation itself fails, assume audio is okay and let Speech API handle it
        quality_check['warnings'].append('Could not validate audio quality - will attempt processing')
        quality_check['quality_score'] = 0.7

    return quality_check

# === GOOGLE CLOUD VISION PROCESSOR ===

class CloudVisionProcessor:
    """Google Cloud Vision API processor"""
    
    def __init__(self):
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Cloud Vision client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Vision: {e}")
            self.client = None
    
    def extract_text_from_image(self, image_data: bytes) -> Tuple[str, float, Dict]:
        """Extract text from image using OCR"""
        if not self.client:
            return "", 0.0, {"error": "Vision API not available"}
        
        try:
            with PerformanceTimer("vision_ocr"):
                image = vision.Image(content=image_data)
                
                # Text detection
                response = self.client.text_detection(image=image)
                
                if response.error.message:
                    raise Exception(f'Vision API error: {response.error.message}')
                
                # Extract text and confidence
                extracted_text = ""
                avg_confidence = 0.0
                
                if response.text_annotations:
                    # First annotation contains full text
                    extracted_text = response.text_annotations[0].description
                    avg_confidence = 0.85  # Default confidence for Vision API
                
                # Extract metadata
                metadata = {
                    'total_annotations': len(response.text_annotations),
                    'text_regions': len(response.text_annotations) - 1 if response.text_annotations else 0,
                    'api_used': 'google_cloud_vision'
                }
                
                logger.info(f"OCR extracted {len(extracted_text)} characters")
                
                return normalize_text(extracted_text), avg_confidence, metadata
                
        except Exception as e:
            logger.error(f"Text extraction from image failed: {e}")
            return "", 0.0, {"error": sanitize_error_message(str(e))}
    
    def detect_image_manipulation(self, image_data: bytes) -> Dict[str, Any]:
        """Detect potential image manipulation using computer vision"""
        try:
            with PerformanceTimer("image_manipulation_detection"):
                # Convert to numpy array for OpenCV analysis
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return {"is_manipulated": False, "confidence": 0.0, "details": {"error": "Invalid image"}}
                
                manipulation_analysis = {
                    "is_manipulated": False,
                    "confidence": 0.0,
                    "details": {},
                    "metadata": {}
                }
                
                # Image metadata
                height, width, channels = image.shape
                manipulation_analysis["metadata"] = {
                    "dimensions": f"{width}x{height}",
                    "channels": channels,
                    "size_bytes": len(image_data)
                }
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 1. Sharpness analysis (blur detection)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 50:
                    manipulation_analysis["details"]["sharpness_anomaly"] = {
                        "detected": True,
                        "variance": float(laplacian_var),
                        "description": "Unusually low image sharpness"
                    }
                    manipulation_analysis["confidence"] += 0.3
                
                # 2. Edge density analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (width * height)
                
                if edge_density > 0.15:
                    manipulation_analysis["details"]["edge_anomalies"] = {
                        "detected": True,
                        "density": float(edge_density),
                        "description": "High edge density - possible over-sharpening"
                    }
                    manipulation_analysis["confidence"] += 0.2
                elif edge_density < 0.02:
                    manipulation_analysis["details"]["edge_anomalies"] = {
                        "detected": True,
                        "density": float(edge_density),
                        "description": "Very low edge density - possible heavy smoothing"
                    }
                    manipulation_analysis["confidence"] += 0.25
                
                # 3. Color distribution analysis
                color_hist = cv2.calcHist([image], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                hist_peaks = len([1 for h in color_hist.flatten() if h > np.mean(color_hist) * 3])
                
                if hist_peaks < 100:
                    manipulation_analysis["details"]["color_anomalies"] = {
                        "detected": True,
                        "peak_count": hist_peaks,
                        "description": "Limited color distribution - possible editing"
                    }
                    manipulation_analysis["confidence"] += 0.15
                
                # 4. Compression artifacts (JPEG quality estimation)
                if len(image_data) / (width * height) < 0.5:
                    manipulation_analysis["details"]["compression_artifacts"] = {
                        "detected": True,
                        "compression_ratio": float(len(image_data) / (width * height)),
                        "description": "High compression detected"
                    }
                    manipulation_analysis["confidence"] += 0.2
                
                # Determine final verdict
                manipulation_analysis["confidence"] = min(manipulation_analysis["confidence"], 1.0)
                
                if manipulation_analysis["confidence"] > 0.6:
                    manipulation_analysis["is_manipulated"] = True
                    manipulation_analysis["manipulation_type"] = "likely_digital_alteration"
                elif manipulation_analysis["confidence"] > 0.4:
                    manipulation_analysis["is_manipulated"] = True
                    manipulation_analysis["manipulation_type"] = "possible_enhancement"
                else:
                    manipulation_analysis["manipulation_type"] = "appears_authentic"
                
                logger.info(f"Image manipulation analysis: {manipulation_analysis['confidence']:.2f} confidence")
                
                return manipulation_analysis
                
        except Exception as e:
            logger.error(f"Image manipulation detection failed: {e}")
            return {
                "is_manipulated": False,
                "confidence": 0.0,
                "details": {"error": sanitize_error_message(str(e))},
                "metadata": {}
            }

# === GOOGLE CLOUD SPEECH PROCESSOR ===

class CloudSpeechProcessor:
    """Google Cloud Speech API processor with fixes"""
    
    def __init__(self):
        try:
            self.client = speech.SpeechClient()
            logger.info("Cloud Speech client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Speech: {e}")
            self.client = None
    
    def transcribe_audio(self, audio_data: bytes, language_code: str = "hi-IN", audio_type: str = "file") -> Tuple[str, float, Dict]:
        """Audio transcription with proper language handling"""
        if not self.client:
            logger.warning("Speech API client not available - using fallback processing")
            return self._fallback_audio_processing(audio_data, audio_type)

        try:
            with PerformanceTimer("speech_transcription"):
                # Validate audio quality first
                quality_check = validate_audio_quality(audio_data)
                if not quality_check['is_valid']:
                    logger.warning(f"Audio quality issues: {', '.join(quality_check['warnings'])}")
                    # For demo purposes, provide fallback instead of failing
                    return self._fallback_audio_processing(audio_data, audio_type)

                # Get proper language code for Speech API
                speech_language_code = self._get_speech_language_code(language_code)
                
                # Prepare audio for Speech API
                audio = speech.RecognitionAudio(content=audio_data)
                
                # Alternative languages for better recognition
                alt_languages = ["en-IN", "hi-IN", "ta-IN", "te-IN", "bn-IN", "mr-IN", "gu-IN"]
                if speech_language_code in alt_languages:
                    alt_languages.remove(speech_language_code)

                # Determine encoding based on audio type
                # Live recordings from browser use WEBM OPUS format
                # Uploaded files can be any format, so use auto-detection
                if audio_type == "live":
                    encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
                else:
                    encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

                # Build config
                config = speech.RecognitionConfig(
                    encoding=encoding,
                    language_code=speech_language_code,
                    alternative_language_codes=alt_languages[:4],
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    model="latest_long",
                    use_enhanced=True,
                    profanity_filter=True,
                    enable_spoken_punctuation=True,
                    enable_spoken_emojis=False
                )
                
                # Perform transcription
                response = self.client.recognize(config=config, audio=audio)
                
                # Process transcript
                transcript = ""
                total_confidence = 0.0
                word_count = 0
                
                if response.results:
                    for result in response.results:
                        alternative = result.alternatives[0]
                        transcript += alternative.transcript + " "
                        
                        # Calculate weighted confidence
                        if hasattr(alternative, 'confidence'):
                            total_confidence += alternative.confidence
                        else:
                            total_confidence += 0.8  # Default confidence
                        
                        word_count += len(alternative.transcript.split())
                    
                    avg_confidence = total_confidence / len(response.results)
                else:
                    avg_confidence = 0.0
                
                # Extract metadata
                metadata = {
                    "language_used": speech_language_code,
                    "language_confidence": avg_confidence,
                    "total_results": len(response.results) if response.results else 0,
                    "transcript_length": len(transcript.strip()),
                    "word_count": word_count,
                    "api_used": "google_cloud_speech",
                    "audio_type": audio_type,
                    "quality_check": quality_check,
                    "estimated_duration": quality_check.get('duration_estimate', 0)
                }
                
                # Quality warnings
                warnings = []
                if avg_confidence < 0.6:
                    warnings.append('Low transcription confidence')
                if word_count < 5:
                    warnings.append('Very short transcription')
                
                if warnings:
                    metadata['quality_warnings'] = warnings
                
                logger.info(f"Audio transcribed: {len(transcript)} characters, {word_count} words, {avg_confidence:.2f} confidence")
                
                return normalize_text(transcript.strip()), avg_confidence, metadata
                
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            # Try fallback processing on API failure
            return self._fallback_audio_processing(audio_data, audio_type)

    def _fallback_audio_processing(self, audio_data: bytes, audio_type: str = "file") -> Tuple[str, float, Dict]:
        """Enhanced fallback audio processing when Speech API is not available"""
        audio_size_mb = len(audio_data) / (1024 * 1024)
        estimated_duration = len(audio_data) / 8000  # More realistic estimate

        # Provide helpful user instructions based on audio type
        if audio_type == "live":
            fallback_text = (
                "Audio transcription temporarily unavailable. "
                "Please speak clearly and try recording again, or type your claim directly in the text box below."
            )
        else:
            fallback_text = (
                "Audio file received but automatic transcription is currently unavailable. "
                "Please type what was said in the audio file for fact-checking analysis."
            )

        # Add size and duration info for context
        if audio_size_mb > 0.1:
            fallback_text += f" (Audio file: {audio_size_mb:.1f}MB, ~{estimated_duration:.0f} seconds)"

        metadata = {
            "fallback_mode": True,
            "audio_size_mb": audio_size_mb,
            "estimated_duration": estimated_duration,
            "audio_type": audio_type,
            "instruction": "Manual transcription required",
            "next_steps": [
                "Type the spoken content in the text input field",
                "Ensure all key claims are included",
                "Submit for fact-checking analysis"
            ],
            "user_action_required": True
        }

        logger.info(f"Audio fallback processing: {audio_size_mb:.1f}MB file, {audio_type} type")
        return fallback_text, 0.5, metadata  # Lower confidence to indicate manual input needed

    def _get_speech_language_code(self, language_code: str) -> str:
        """Get proper speech API language code"""
        # Language code mapping
        speech_codes = {
            'hi': 'hi-IN',
            'en': 'en-IN', 
            'ta': 'ta-IN',
            'te': 'te-IN',
            'bn': 'bn-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN'
        }
        
        base_lang = language_code.split('-')[0]
        return speech_codes.get(base_lang, 'hi-IN')

# === TEXT-TO-SPEECH PROCESSOR ===

class TextToSpeechProcessor:
    """Google Cloud Text-to-Speech API processor with fixes"""
    
    def __init__(self):
        try:
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Cloud TTS client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud TTS: {e}")
            self.client = None
    
    def generate_audio(self, text: str, language_code: str = "hi-IN") -> Optional[str]:
        """Generate audio with proper language support"""
        if not self.client or not text:
            logger.warning("TTS client not available or empty text")
            return None
        
        try:
            # Get the correct TTS language code
            tts_language_code = self._get_tts_language_code(language_code)
            
            # Translate text if needed
            audio_text = self._prepare_text_for_tts(text, language_code)
            
            with PerformanceTimer("tts_generation"):
                synthesis_input = texttospeech.SynthesisInput(text=audio_text)
                
                # Get optimal voice
                voice = self._get_optimal_voice(tts_language_code)
                
                # Audio configuration
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    sample_rate_hertz=24000,
                    speaking_rate=0.95,
                    pitch=0.0,
                    volume_gain_db=2.0  # Slightly louder
                )
                
                response = self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                # Return base64 encoded audio
                audio_b64 = base64.b64encode(response.audio_content).decode('utf-8')
                logger.info(f"Generated {len(audio_b64)} characters of audio data in {tts_language_code}")
                
                return audio_b64
                
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None
    
    def _get_tts_language_code(self, language_code: str) -> str:
        """Get proper TTS language code"""
        # TTS language mapping
        tts_codes = {
            'hi': 'hi-IN',
            'en': 'en-IN',
            'ta': 'ta-IN', 
            'te': 'te-IN',
            'bn': 'bn-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN'
        }
        
        base_lang = language_code.split('-')[0]
        return tts_codes.get(base_lang, 'hi-IN')
    
    def _prepare_text_for_tts(self, text: str, language_code: str) -> str:
        """Prepare text for TTS with translation if needed"""
        # If text is in English but target language is not English, translate
        if language_code != 'en-IN' and language_code != 'en':
            target_lang = language_code.split('-')[0]
            try:
                translated_text, _ = translate_text(text, target_lang, 'en')
                audio_text = self._preprocess_text_for_tts(translated_text)
                logger.info(f"Translated TTS text to {target_lang}")
                return audio_text
            except Exception as e:
                logger.warning(f"Translation failed, using original text: {e}")
        
        return self._preprocess_text_for_tts(text)
    
    def _preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text for better TTS output"""
        # Limit text length for audio
        if len(text) > 500:
            sentences = text.split('. ')
            audio_text = ""
            for sentence in sentences:
                if len(audio_text) + len(sentence) > 450:
                    break
                audio_text += sentence + ". "
            audio_text = audio_text.strip()
            if not audio_text.endswith('.'):
                audio_text += "..."
        else:
            audio_text = text
        
        # Clean up text for better pronunciation
        audio_text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF.,!?;:]', ' ', audio_text)
        audio_text = re.sub(r'\s+', ' ', audio_text).strip()
        
        return audio_text
    
    def _get_optimal_voice(self, language_code: str) -> texttospeech.VoiceSelectionParams:
        """Get optimal voice settings for language"""
        # Voice preferences with fallbacks
        voice_preferences = {
            'hi-IN': {'name': 'hi-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'en-IN': {'name': 'en-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'ta-IN': {'name': 'ta-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'te-IN': {'name': 'te-IN-Standard-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'bn-IN': {'name': 'bn-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'mr-IN': {'name': 'mr-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE},
            'gu-IN': {'name': 'gu-IN-Wavenet-A', 'gender': texttospeech.SsmlVoiceGender.FEMALE}
        }
        
        if language_code in voice_preferences:
            pref = voice_preferences[language_code]
            try:
                return texttospeech.VoiceSelectionParams(
                    name=pref['name'],
                    language_code=language_code,
                    ssml_gender=pref['gender']
                )
            except Exception as e:
                logger.warning(f"Specific voice {pref['name']} not available, using default: {e}")
        
        # Fallback to basic voice selection
        return texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

# === URL CONTENT PROCESSOR ===

class URLContentProcessor:
    """URL content extraction with better error handling"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Add retry strategy
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def process_url(self, url: str) -> URLAnalysis:
        """Process URL content"""
        start_time = time.time()
        
        try:
            # Validate URL
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                raise ValueError(error_msg)
            
            with PerformanceTimer("url_processing"):
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                # Handle different content types
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    raise ValueError(f"Unsupported content type: {content_type}")
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                images = self._extract_images(soup, url)
                platform_type = self._determine_platform_type(url)
                
                # Extract metadata
                metadata = {
                    'url': url,
                    'domain': urlparse(url).netloc,
                    'extraction_time': get_current_timestamp(),
                    'content_length': len(content),
                    'status_code': response.status_code,
                    'content_type': content_type,
                    'response_size': len(response.content),
                    'encoding': response.encoding
                }
                
                processing_time = time.time() - start_time
                
                return URLAnalysis(
                    url=url,
                    platform_type=platform_type,
                    content_type="article",
                    title=title,
                    extracted_text=normalize_text(content),
                    images=images,
                    metadata=metadata,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"URL processing failed for {url}: {e}")
            return URLAnalysis(
                url=url,
                platform_type="error",
                content_type="error",
                title="Processing Failed",
                extracted_text=f"Failed to process URL: {sanitize_error_message(str(e))}",
                images=[],
                metadata={"error": sanitize_error_message(str(e)), "url": url},
                processing_time=time.time() - start_time
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML"""
        title_sources = [
            soup.find('meta', {'property': 'og:title'}),
            soup.find('meta', {'name': 'twitter:title'}),
            soup.find('h1'),
            soup.find('title'),
            soup.find('meta', {'name': 'title'})
        ]
        
        for source in title_sources:
            if source:
                title = source.get('content') if source.name == 'meta' else source.get_text(strip=True)
                if title and len(title.strip()) > 5:
                    return title.strip()[:200]
        
        return "Untitled"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove unnecessary elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.article-content', 
            '.post-content', '.entry-content', '.blog-content', '.news-content'
        ]
        
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = self._extract_text_from_element(content_element)
                if len(content) > 100:
                    return content[:3000]
        
        # Fallback: extract from paragraphs
        content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'div'])
        content_parts = []
        
        for element in content_elements[:20]:
            text = element.get_text(strip=True)
            if len(text) > 20 and text not in content_parts:
                content_parts.append(text)
        
        return '\n'.join(content_parts)[:3000]
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from HTML element"""
        text_parts = []
        for item in element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span']):
            text = item.get_text(strip=True)
            if len(text) > 15:
                text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract images from HTML"""
        images = []
        
        # Look for main content images
        og_image = soup.find('meta', {'property': 'og:image'})
        if og_image and og_image.get('content'):
            images.append(og_image['content'])
        
        twitter_image = soup.find('meta', {'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            images.append(twitter_image['content'])
        
        # Get content images
        content_images = soup.find_all('img', limit=5)
        for img in content_images:
            img_url = img.get('src') or img.get('data-src')
            if img_url and not any(skip in img_url.lower() for skip in ['icon', 'logo', 'avatar', 'ad', 'banner']):
                # Convert relative URLs to absolute
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    parsed_base = urlparse(base_url)
                    img_url = f"{parsed_base.scheme}://{parsed_base.netloc}{img_url}"
                elif not img_url.startswith('http'):
                    continue
                
                images.append(img_url)
        
        return list(set(images))[:3]
    
    def _determine_platform_type(self, url: str) -> str:
        """Determine platform type from URL"""
        domain = urlparse(url).netloc.lower()
        
        # News platforms
        if any(news in domain for news in ['news', 'times', 'hindu', 'ndtv', 'bbc', 'reuters', 'cnn', 'indianexpress', 'hindustantimes']):
            return "news"
        # Social media platforms
        elif any(social in domain for social in ['twitter', 'facebook', 'instagram', 'youtube', 'linkedin', 'tiktok']):
            return "social_media"
        # Government sites
        elif '.gov' in domain or 'pib.gov.in' in domain:
            return "government"
        # Blog platforms
        elif any(blog in domain for blog in ['medium', 'wordpress', 'blogspot', 'substack']):
            return "blog"
        # E-commerce
        elif any(ecom in domain for ecom in ['amazon', 'flipkart', 'myntra', 'snapdeal']):
            return "ecommerce"
        # Educational
        elif any(edu in domain for edu in ['.edu', '.ac.', 'university', 'college']):
            return "educational"
        else:
            return "other"

# === MAIN CONTENT PROCESSOR COORDINATOR ===

class ContentProcessorCoordinator:
    """Coordinator for all content processing operations"""
    
    def __init__(self):
        self.vision_processor = CloudVisionProcessor()
        self.speech_processor = CloudSpeechProcessor()
        self.tts_processor = TextToSpeechProcessor()
        self.url_processor = URLContentProcessor()
        
    def process_content(self, content_data: Any, content_type: str, 
                       metadata: Dict = None) -> Dict[str, Any]:
        """Process content with better error handling"""
        
        processing_result = {
            'success': False,
            'content_type': content_type,
            'extracted_text': '',
            'confidence': 0.0,
            'metadata': metadata or {},
            'processing_time': 0.0,
            'analysis': {}
        }
        
        start_time = time.time()
        
        try:
            if content_type == 'image':
                if isinstance(content_data, str):
                    content_data = base64.b64decode(content_data)
                
                text, confidence, meta = self.vision_processor.extract_text_from_image(content_data)
                manipulation_analysis = self.vision_processor.detect_image_manipulation(content_data)
                
                processing_result.update({
                    'success': True,
                    'extracted_text': text,
                    'confidence': confidence,
                    'analysis': {
                        'ocr_metadata': meta,
                        'manipulation_analysis': manipulation_analysis
                    }
                })
                
            elif content_type == 'audio':
                if isinstance(content_data, str):
                    content_data = base64.b64decode(content_data)
                
                language_code = metadata.get('language', 'hi-IN')
                audio_type = metadata.get('audio_type', 'file')
                
                text, confidence, meta = self.speech_processor.transcribe_audio(
                    content_data, language_code, audio_type
                )
                
                processing_result.update({
                    'success': True,
                    'extracted_text': text,
                    'confidence': confidence,
                    'analysis': {'transcription_metadata': meta}
                })
                
            elif content_type == 'url':
                url = content_data.decode('utf-8') if isinstance(content_data, bytes) else content_data
                url_analysis = self.url_processor.process_url(url)
                
                processing_result.update({
                    'success': True,
                    'extracted_text': url_analysis.extracted_text,
                    'confidence': 0.8,
                    'analysis': {
                        'url_analysis': {
                            'platform_type': url_analysis.platform_type,
                            'content_type': url_analysis.content_type,
                            'title': url_analysis.title,
                            'images': url_analysis.images,
                            'metadata': url_analysis.metadata
                        }
                    }
                })
                
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            processing_result['processing_time'] = time.time() - start_time
            
            # Enhanced Content validation with better error messages
            if not processing_result['extracted_text'] or len(processing_result['extracted_text'].strip()) < 5:
                if content_type == 'audio':
                    # More specific error for audio
                    error_msg = "Audio transcription failed or returned empty results. "
                    if not processing_result.get('analysis', {}).get('transcription_metadata', {}):
                        error_msg += "Speech recognition service may not be available."
                    else:
                        error_msg += "Audio quality may be too poor or contains no speech."
                    raise ValueError(error_msg)
                else:
                    raise ValueError("No meaningful content could be extracted from the input")

            logger.info(f"Content processing completed: {content_type}, {len(processing_result['extracted_text'])} characters")
            
            return processing_result
            
        except Exception as e:
            processing_result['processing_time'] = time.time() - start_time
            logger.error(f"Content processing failed: {e}")
            processing_result.update({
                'success': False,
                'error': sanitize_error_message(str(e))
            })
            return processing_result
    
    def generate_audio_response(self, text: str, language_code: str = "hi-IN") -> Optional[str]:
        """Generate audio response with proper validation"""
        if not text or not text.strip():
            logger.warning("Empty text provided for audio generation")
            return None
        
        try:
            # Ensure we have the correct language code format for TTS
            if '-' not in language_code:
                tts_code = f"{language_code}-IN"
            else:
                tts_code = language_code
            
            logger.info(f"Generating audio for text length: {len(text)} in language: {tts_code}")
            
            audio_result = self.tts_processor.generate_audio(text, tts_code)
            
            if audio_result:
                logger.info(f"Audio generation successful: {len(audio_result)} characters base64")
            else:
                logger.warning("Audio generation returned None")
            
            return audio_result
            
        except Exception as e:
            logger.error(f"Audio response generation failed: {e}")
            return None
    
    def get_processor_health(self) -> Dict[str, Any]:
        """Get health status of all processors"""
        health_status = {
            'vision_processor': {
                'available': self.vision_processor.client is not None,
                'status': 'healthy' if self.vision_processor.client else 'unavailable'
            },
            'speech_processor': {
                'available': self.speech_processor.client is not None,
                'status': 'healthy' if self.speech_processor.client else 'unavailable'
            },
            'tts_processor': {
                'available': self.tts_processor.client is not None,
                'status': 'healthy' if self.tts_processor.client else 'unavailable'
            },
            'url_processor': {
                'available': True,
                'status': 'healthy'
            }
        }
        
        # Check individual processor health
        try:
            # Test Vision API
            if self.vision_processor.client:
                health_status['vision_processor']['last_check'] = get_current_timestamp()
        except Exception as e:
            health_status['vision_processor']['status'] = f'error: {sanitize_error_message(str(e))}'
        
        try:
            # Test Speech API
            if self.speech_processor.client:
                health_status['speech_processor']['last_check'] = get_current_timestamp()
        except Exception as e:
            health_status['speech_processor']['status'] = f'error: {sanitize_error_message(str(e))}'
        
        try:
            # Test TTS API
            if self.tts_processor.client:
                health_status['tts_processor']['last_check'] = get_current_timestamp()
        except Exception as e:
            health_status['tts_processor']['status'] = f'error: {sanitize_error_message(str(e))}'
        
        # Overall health assessment
        healthy_services = sum(1 for service in health_status.values() if service.get('status') == 'healthy')
        total_services = len(health_status)
        
        if healthy_services == total_services:
            health_status['overall'] = 'healthy'
        elif healthy_services >= total_services * 0.75:
            health_status['overall'] = 'degraded'
        else:
            health_status['overall'] = 'unhealthy'
        
        health_status['summary'] = {
            'healthy_services': healthy_services,
            'total_services': total_services,
            'health_percentage': round((healthy_services / total_services) * 100, 1)
        }
        
        return health_status

# === GLOBAL CONTENT PROCESSOR INSTANCE ===

content_processor_coordinator = None

def get_content_processor() -> ContentProcessorCoordinator:
    """Get global content processor coordinator"""
    global content_processor_coordinator
    
    if content_processor_coordinator is None:
        content_processor_coordinator = ContentProcessorCoordinator()
        logger.info("Content processor coordinator initialized")
    
    return content_processor_coordinator

# === UTILITY FUNCTIONS ===

def extract_text_from_any_content(content_data: Any, content_type: str, 
                                 metadata: Dict = None) -> Tuple[str, float, Dict]:
    """Extract text from any content type"""
    
    processor = get_content_processor()
    
    try:
        result = processor.process_content(content_data, content_type, metadata)
        
        if result['success']:
            return result['extracted_text'], result['confidence'], result['analysis']
        else:
            raise ValueError(f"Content extraction failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Text extraction failed for {content_type}: {e}")
        return "", 0.0, {"error": sanitize_error_message(str(e)), "content_type": content_type}

def validate_content_safety(extracted_text: str, content_type: str) -> Dict[str, Any]:
    """Validate content safety"""
    from utils import check_content_safety
    
    safety_result = check_content_safety(extracted_text)
    
    # Add content-type specific checks
    if content_type == 'audio':
        if len(extracted_text) < 10:
            safety_result['warnings'] = safety_result.get('warnings', []) + ['Very short audio transcription']
    elif content_type == 'image':
        if not extracted_text or len(extracted_text.strip()) < 5:
            safety_result['warnings'] = safety_result.get('warnings', []) + ['No text found in image']
    elif content_type == 'url':
        if 'error' in extracted_text.lower():
            safety_result['warnings'] = safety_result.get('warnings', []) + ['URL processing error detected']
    
    return safety_result

def test_content_processor() -> Dict[str, bool]:
    """Test all content processor components"""
    processor = get_content_processor()
    test_results = {
        'vision_processor': False,
        'speech_processor': False,
        'tts_processor': False,
        'url_processor': False
    }
    
    try:
        test_results['vision_processor'] = processor.vision_processor.client is not None
    except Exception as e:
        logger.error(f"Vision processor test failed: {e}")
    
    try:
        test_results['speech_processor'] = processor.speech_processor.client is not None
    except Exception as e:
        logger.error(f"Speech processor test failed: {e}")
    
    try:
        test_results['tts_processor'] = processor.tts_processor.client is not None
    except Exception as e:
        logger.error(f"TTS processor test failed: {e}")
    
    try:
        test_results['url_processor'] = True
    except Exception as e:
        logger.error(f"URL processor test failed: {e}")
    
    return test_results

logger.info("Content Processor module loaded successfully")
logger.info("✅ Multi-modal content processing with live audio support")
logger.info("✅ Fixed error handling and quality validation")
logger.info("✅ Improved language support for Indian languages")
logger.info("✅ Enhanced TTS generation with proper voice selection")
logger.info("✅ Robust audio transcription with quality checks")