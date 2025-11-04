"""
SatyaCheck AI - WhatsApp Bot Integration
Twilio WhatsApp API for multi-media fact-checking
"""

import logging
import os
import tempfile
import requests
from flask import Blueprint, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from typing import Optional, Dict, Any
import threading

from main import search_pib_factcheck
from ai_analyzer import analyze_claim_with_ai
from content_processor import get_content_processor
from utils import detect_language_heuristic, generate_request_id, validate_input_text

logger = logging.getLogger(__name__)

whatsapp_bp = Blueprint('whatsapp', __name__)

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio WhatsApp client initialized")
else:
    logger.warning("Twilio credentials not found. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")


def whatsapp_webhook(request=None):
    """Main WhatsApp webhook - handles all message types"""
    if request is None:
        from flask import request
    try:
        from_number = request.form.get('From')
        message_body = request.form.get('Body', '').strip()
        num_media = int(request.form.get('NumMedia', 0))

        logger.info(f"WhatsApp from {from_number}: {message_body[:50]}... | Media: {num_media}")

        request_id = generate_request_id()

        if num_media > 0:
            language = _detect_user_language(message_body) if message_body else 'en'
            return _handle_media_message(from_number, message_body, num_media, language, request_id)

        elif 'http://' in message_body or 'https://' in message_body:
            language = _detect_user_language(message_body)
            return _handle_url_message(from_number, message_body, language, request_id)

        elif message_body:
            language = _detect_user_language(message_body)
            return _handle_text_message(from_number, message_body, language, request_id)

        else:
            return _send_welcome_message(from_number, 'en')

    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}", exc_info=True)
        return _send_error_message(request.form.get('From'), str(e))


def _handle_media_message(from_number: str, caption: str, num_media: int, language: str, request_id: str):
    """Handle media files (images, audio, video)"""

    response = MessagingResponse()

    response.message(_get_text('processing_media', language))

    media_items = []
    for i in range(num_media):
        media_items.append({
            'url': request.form.get(f'MediaUrl{i}'),
            'type': request.form.get(f'MediaContentType{i}', '')
        })

    thread = threading.Thread(
        target=_process_media_async,
        args=(from_number, caption, media_items, language, request_id)
    )
    thread.daemon = True
    thread.start()

    headers = {'Content-Type': 'application/xml'}
    return (str(response), 200, headers)


def _process_media_async(from_number: str, caption: str, media_items: list, language: str, request_id: str):
    """Process media files asynchronously"""
    try:
        for i, media_item in enumerate(media_items):
            media_url = media_item['url']
            media_type = media_item['type']

            logger.info(f"Processing media {i+1}/{len(media_items)}: {media_type}")

            media_data = requests.get(media_url, timeout=30).content

            if 'image' in media_type:
                result = _process_image(media_data, caption, language)
            elif 'audio' in media_type:
                result = _process_audio(media_data, caption, language)
            elif 'video' in media_type:
                result = _process_video(media_data, caption, language)
            else:
                result = {'error': 'Unsupported media type'}

            _send_whatsapp_message(from_number, _format_result(result, language))

    except Exception as e:
        logger.error(f"Async media processing error: {e}", exc_info=True)
        _send_whatsapp_message(from_number, _get_text('analysis_failed', language))


def _process_image(image_data: bytes, caption: str, language: str) -> Dict[str, Any]:
    """Process image: Extract text via OCR and fact-check"""
    try:
        processor = get_content_processor()

        logger.info(f"Processing image: {len(image_data)} bytes, caption: {caption[:50] if caption else 'None'}")

        result = processor.process_content(image_data, 'image')

        logger.info(f"Image processing result: success={result.get('success')}, extracted_length={len(result.get('extracted_text', ''))}")

        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Image processing failed: {error_msg}")
            if caption and caption.strip():
                logger.info(f"Using caption as fallback: {caption}")
                return _analyze_claim(caption, language)
            return {'error': 'Could not extract text from image. Please send an image with clear text or add a caption with the claim.'}

        extracted_text = result.get('extracted_text', '').strip()

        if caption and extracted_text:
            full_text = f"{caption}\n\n{extracted_text}"
        elif extracted_text:
            full_text = extracted_text
        elif caption:
            full_text = caption
        else:
            return {'error': 'No text found in image. Please send an image with text content or add a caption.'}

        return _analyze_claim(full_text, language)

    except Exception as e:
        logger.error(f"Image processing exception: {e}", exc_info=True)
        if caption and caption.strip():
            logger.info(f"Using caption as fallback after exception: {caption}")
            return _analyze_claim(caption, language)
        return {'error': f'Image processing failed. Please try again or send the text directly.'}


def _process_audio(audio_data: bytes, caption: str, language: str) -> Dict[str, Any]:
    """Process audio: Transcribe to text and fact-check"""
    try:
        language_map = {
            'hi': 'hi-IN',
            'en': 'en-US',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'bn': 'bn-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN'
        }
        language_code = language_map.get(language, 'en-US')

        processor = get_content_processor()
        result = processor.process_content(audio_data, 'audio', {'language': language_code})

        if not result.get('success'):
            return {'error': 'Failed to transcribe audio. Please ensure audio is clear with speech.'}

        transcription = result.get('extracted_text', '').strip()
        if not transcription:
            return {'error': 'No speech detected in audio. Please send clear audio with spoken content.'}

        return _analyze_claim(transcription, language)

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {'error': f'Audio processing failed: {str(e)}'}


def _process_video(video_data: bytes, caption: str, language: str) -> Dict[str, Any]:
    """Process video: Extract audio → Transcribe to text → Fact-check"""
    try:
        import subprocess

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_file:
            video_file.write(video_data)
            video_path = video_file.name

        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, capture_output=True, timeout=60)

            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            language_map = {
                'hi': 'hi-IN',
                'en': 'en-US',
                'ta': 'ta-IN',
                'te': 'te-IN',
                'bn': 'bn-IN',
                'mr': 'mr-IN',
                'gu': 'gu-IN'
            }
            language_code = language_map.get(language, 'en-US')

            processor = get_content_processor()
            result = processor.process_content(audio_data, 'audio', {'language': language_code})

            if not result.get('success'):
                return {'error': 'Failed to transcribe video audio. Please ensure video has clear speech.'}

            transcription = result.get('extracted_text', '').strip()
            if not transcription:
                return {'error': 'No speech detected in video. Please send video with spoken content.'}

            return _analyze_claim(transcription, language)

        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except subprocess.TimeoutExpired:
        logger.error("Video processing timeout")
        return {'error': 'Video processing took too long. Please send shorter videos.'}
    except FileNotFoundError:
        logger.error("ffmpeg not found")
        return {'error': 'Video processing unavailable. Please send audio or text instead.'}
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {'error': f'Video processing failed: {str(e)}'}


def _handle_url_message(from_number: str, message_body: str, language: str, request_id: str):
    """Handle messages containing URLs"""
    response = MessagingResponse()
    response.message(_get_text('processing_url', language))

    thread = threading.Thread(
        target=_process_url_async,
        args=(from_number, message_body, language, request_id)
    )
    thread.daemon = True
    thread.start()

    headers = {'Content-Type': 'application/xml'}
    return (str(response), 200, headers)


def _process_url_async(from_number: str, message_body: str, language: str, request_id: str):
    """Process URL content asynchronously"""
    try:
        import re
        urls = re.findall(r'https?://[^\s]+', message_body)

        if not urls:
            _send_whatsapp_message(from_number, _get_text('no_url_found', language))
            return

        url = urls[0]

        processor = get_content_processor()
        result = processor.process_content(url, 'url')

        if not result.get('success'):
            _send_whatsapp_message(from_number, _get_text('url_extraction_failed', language))
            return

        content = result.get('extracted_text', '')
        if not content:
            _send_whatsapp_message(from_number, _get_text('url_extraction_failed', language))
            return

        analysis_result = _analyze_claim(content, language)
        analysis_result['url'] = url

        _send_whatsapp_message(from_number, _format_result(analysis_result, language))

    except Exception as e:
        logger.error(f"URL processing error: {e}")
        _send_whatsapp_message(from_number, _get_text('analysis_failed', language))


def _handle_text_message(from_number: str, message_body: str, language: str, request_id: str):
    """Handle plain text fact-checking"""
    response = MessagingResponse()

    if message_body.lower() in ['hi', 'hello', 'help', 'start', 'namaste']:
        return _send_welcome_message(from_number, language)

    is_valid, error_msg = validate_input_text(message_body)
    if not is_valid:
        response.message(error_msg or _get_text('invalid_input', language))
        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)

    response.message(_get_text('processing_text', language))

    thread = threading.Thread(
        target=_analyze_and_send,
        args=(from_number, message_body, language, request_id)
    )
    thread.daemon = True
    thread.start()

    headers = {'Content-Type': 'application/xml'}
    return (str(response), 200, headers)


def _analyze_and_send(from_number: str, claim: str, language: str, request_id: str):
    """Analyze claim and send result"""
    try:
        result = _analyze_claim(claim, language)
        formatted_result = _format_result(result, language)
        _send_whatsapp_message(from_number, formatted_result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        _send_whatsapp_message(from_number, _get_text('analysis_failed', language))


def _analyze_claim(claim: str, language: str) -> Dict[str, Any]:
    """Reuse existing fact-checking pipeline"""
    try:
        pib_results = search_pib_factcheck(claim)

        ai_result = analyze_claim_with_ai(
            claim=claim,
            pib_results=pib_results,
            target_language=language
        )

        return {
            'type': 'fact_check',
            'verdict': ai_result.get('verdict', 'Unverifiable'),
            'confidence': ai_result.get('score', 0),
            'explanation': ai_result.get('explanation', ''),
            'sources': ai_result.get('sources', [])[:3],
            'claim': claim[:100]
        }

    except Exception as e:
        logger.error(f"Claim analysis error: {e}")
        return {'error': str(e)}


def _format_result(result: Dict[str, Any], language: str) -> str:
    """Format fact-check result for WhatsApp"""

    if result.get('error'):
        return f"Error: {result['error']}"

    verdict = result.get('verdict', 'Unverifiable')
    confidence = result.get('confidence', 0)
    explanation = result.get('explanation', '')
    sources = result.get('sources', [])

    if confidence > 1:
        confidence = confidence / 100.0

    message = f"*{verdict}*\n"
    message += f"Confidence: {confidence:.0%}\n\n"

    if explanation:
        message += f"*Explanation:*\n{explanation}\n\n"

    if sources:
        message += f"*Sources:*\n"
        for i, source in enumerate(sources[:3], 1):
            if isinstance(source, dict):
                source_name = source.get('source', source.get('title', 'Unknown'))
                source_url = source.get('url', '')
            else:
                source_name = str(source)
                source_url = ''

            if source_url and len(source_url) < 60:
                message += f"{i}. {source_name}\n   {source_url}\n"
            else:
                message += f"{i}. {source_name}\n"

    message += f"\n_Powered by SatyaCheck AI_"

    return message


def _send_whatsapp_message(to_number: str, message: str):
    """Send WhatsApp message using Twilio"""
    if not twilio_client:
        logger.warning("Twilio client not initialized")
        return

    try:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to_number
        )
        logger.info(f"Sent WhatsApp message to {to_number}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")


def _send_welcome_message(from_number: str, language: str):
    """Send welcome message"""
    response = MessagingResponse()

    welcome_msg = """*Welcome to SatyaCheck AI!*

*I can fact-check:*
Text messages
Images (OCR text extraction)
Audio (speech transcription)
Videos (audio extraction + transcription)
URLs (content analysis)

*How to use:*
Simply send me any claim, screenshot, voice note, video, or link - I'll extract the text and verify it with sources!

*Multi-language support:*
Hindi, English, Tamil, Telugu, Bengali, Marathi, Gujarati

_Your response will be in the same language as your message!_

*Powered by SatyaCheck AI*"""

    response.message(welcome_msg)

    headers = {'Content-Type': 'application/xml'}
    return (str(response), 200, headers)


def _send_error_message(from_number: str, error: str):
    """Send error message"""
    response = MessagingResponse()
    response.message(f"Error: {error}\n\nType 'help' for assistance.")
    headers = {'Content-Type': 'application/xml'}
    return (str(response), 200, headers)


def _detect_user_language(text: str) -> str:
    """Detect user's language"""
    if not text:
        return 'en'
    return detect_language_heuristic(text)


def _get_text(key: str, language: str) -> str:
    """Get localized text"""
    texts = {
        'processing_media': {
            'en': 'Processing your media... This may take a moment.',
            'hi': 'आपकी मीडिया प्रोसेस हो रही है... कृपया प्रतीक्षा करें।'
        },
        'processing_url': {
            'en': 'Analyzing the URL... Please wait.',
            'hi': 'URL का विश्लेषण हो रहा है... कृपया प्रतीक्षा करें।'
        },
        'processing_text': {
            'en': 'Fact-checking your claim... Please wait.',
            'hi': 'आपके दावे की जांच हो रही है... कृपया प्रतीक्षा करें।'
        },
        'analysis_failed': {
            'en': 'Analysis failed. Please try again or send a different message.',
            'hi': 'विश्लेषण विफल रहा। कृपया पुनः प्रयास करें।'
        },
        'invalid_input': {
            'en': 'Invalid input. Please send a valid claim, image, audio, or video.',
            'hi': 'अमान्य इनपुट। कृपया वैध संदेश भेजें।'
        },
        'no_url_found': {
            'en': 'No URL found in your message.',
            'hi': 'संदेश में कोई URL नहीं मिला।'
        },
        'url_extraction_failed': {
            'en': 'Could not extract content from URL.',
            'hi': 'URL से सामग्री निकालने में विफल।'
        },
        'error': {
            'en': 'Error',
            'hi': 'त्रुटि'
        }
    }

    return texts.get(key, {}).get(language, texts.get(key, {}).get('en', ''))


logger.info("WhatsApp bot module initialized")
