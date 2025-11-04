"""
SatyaCheck AI - Advanced IVR System
Gemini 2.0 powered conversational IVR with real-time fact-checking
"""

import logging
import os
import json
import time
from flask import Blueprint, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Stream
from twilio.rest import Client
from typing import Dict, Optional, List, Any
import threading
from datetime import datetime, timedelta
import asyncio

from main import search_pib_factcheck, search_web
from ai_analyzer import analyze_claim_with_ai
from content_processor import get_content_processor
from utils import detect_language_heuristic, generate_request_id
from database import save_analysis_result, get_storage
from gemini_service import get_gemini_service

logger = logging.getLogger(__name__)

ivr_advanced_bp = Blueprint('ivr_advanced', __name__)

try:
    import redis
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True,
        socket_connect_timeout=2
    )
    redis_available = redis_client.ping()
    logger.info("Redis connected for IVR sessions")
except:
    redis_client = None
    redis_available = False
    logger.warning("Redis not available, using in-memory sessions")

ivr_sessions = {}
ivr_analytics = {
    'total_calls': 0,
    'claims_verified': 0,
    'fake_detected': 0,
    'languages_used': {},
    'peak_hours': {},
    'regional_patterns': {}
}


@ivr_advanced_bp.route('/ivr/v2/incoming', methods=['POST'])
def handle_incoming_call_v2():
    """Next-gen IVR entry point with natural conversation"""
    try:
        caller_number = request.form.get('From')
        call_sid = request.form.get('CallSid')

        logger.info(f"ADVANCED IVR: Call from {caller_number}")

        _track_call_analytics(caller_number, call_sid)

        session = {
            'caller': caller_number,
            'call_sid': call_sid,
            'language': 'auto',
            'mode': 'conversational',
            'conversation_history': [],
            'claims_checked': 0,
            'start_time': time.time(),
            'sentiment': 'neutral',
            'trust_score': 100
        }

        _save_session(call_sid, session)

        return _conversational_greeting(call_sid)

    except Exception as e:
        logger.error(f"Advanced IVR error: {e}", exc_info=True)
        return _error_response()


def _conversational_greeting(call_sid: str) -> Response:
    """Natural conversation greeting"""
    response = VoiceResponse()

    greeting = (
        "Namaste! Welcome to Satya Check A I. "
        "I'm your A I fact-checker. "
        "Just tell me any claim you want to verify, "
        "and I'll check it instantly using Google Gemini. "
        "You can speak in Hindi, English, or mix both languages. "
        "What would you like me to check?"
    )

    response.say(greeting, language='en-IN', voice='Polly.Aditi')

    response.record(
        max_length=60,
        transcribe=True,
        transcribe_callback=f'/ivr/v2/analyze-realtime?call_sid={call_sid}',
        play_beep=True,
        timeout=3,
        finish_on_key='#'
    )

    response.say(
        "I didn't hear anything. Let me repeat.",
        language='en-IN'
    )
    response.redirect('/ivr/v2/incoming')

    return Response(str(response), mimetype='application/xml')


@ivr_advanced_bp.route('/ivr/v2/analyze-realtime', methods=['POST'])
def analyze_claim_realtime():
    """Instant analysis during the call"""
    try:
        call_sid = request.args.get('call_sid') or request.form.get('CallSid')
        transcription = request.form.get('TranscriptionText', '')
        recording_url = request.form.get('RecordingUrl', '')

        session = _get_session(call_sid)

        logger.info(f"Real-time analysis: {transcription[:100]}")

        response = VoiceResponse()
        response.say(
            "Let me check that for you using Google Gemini A I.",
            language='en-IN',
            voice='Polly.Aditi'
        )

        if not transcription:
            transcription = _transcribe_with_enhanced_google(recording_url, 'auto')

        if not transcription or len(transcription.strip()) < 5:
            response.say(
                "I couldn't understand that clearly. Could you please repeat?",
                language='en-IN'
            )
            response.redirect('/ivr/v2/incoming')
            return Response(str(response), mimetype='application/xml')

        language = detect_language_heuristic(transcription)
        session['language'] = language

        deepfake_risk = _analyze_voice_authenticity(recording_url, session)
        if deepfake_risk > 0.7:
            response.say(
                "Warning: This audio may contain voice manipulation. "
                "Analyzing the claim anyway.",
                language='en-IN'
            )

        result = _instant_fact_check(transcription, language, session)

        session['claims_checked'] += 1
        session['conversation_history'].append({
            'claim': transcription,
            'verdict': result.get('verdict'),
            'timestamp': time.time()
        })
        _save_session(call_sid, session)

        voice_result = _format_voice_result_v2(result, language, deepfake_risk)
        response.say(voice_result, language=_get_tts_language(language), voice='Polly.Aditi')

        response.pause(length=1)
        response.say(
            "Would you like to check another claim? Say yes or just speak your claim. "
            "Say no or goodbye to end the call.",
            language='en-IN'
        )

        response.record(
            max_length=60,
            transcribe=True,
            transcribe_callback=f'/ivr/v2/continue?call_sid={call_sid}',
            timeout=5,
            finish_on_key='#'
        )

        threading.Thread(
            target=_send_sms_summary,
            args=(session['caller'], result, transcription),
            daemon=True
        ).start()

        return Response(str(response), mimetype='application/xml')

    except Exception as e:
        logger.error(f"Real-time analysis error: {e}", exc_info=True)
        return _error_response()


@ivr_advanced_bp.route('/ivr/v2/continue', methods=['POST'])
def continue_conversation():
    """Handle multi-turn conversation"""
    try:
        call_sid = request.args.get('call_sid') or request.form.get('CallSid')
        transcription = request.form.get('TranscriptionText', '')
        recording_url = request.form.get('RecordingUrl', '')

        session = _get_session(call_sid)

        if not transcription:
            transcription = _transcribe_with_enhanced_google(recording_url, 'auto')

        transcription_lower = transcription.lower()

        if any(word in transcription_lower for word in ['no', 'bye', 'goodbye', 'end', 'nahi', 'alvida']):
            return _end_call_with_summary(call_sid, session)

        if any(word in transcription_lower for word in ['yes', 'ha', 'haan', 'check', 'verify']):
            response = VoiceResponse()
            response.say(
                "Great! What would you like me to verify?",
                language='en-IN',
                voice='Polly.Aditi'
            )
            response.redirect(f'/ivr/v2/analyze-realtime?call_sid={call_sid}')
            return Response(str(response), mimetype='application/xml')

        if len(transcription.strip()) > 10:
            response = VoiceResponse()
            response.redirect(f'/ivr/v2/analyze-realtime?call_sid={call_sid}')
            return Response(str(response), mimetype='application/xml')

        response = VoiceResponse()
        response.say(
            "I didn't catch that. Say yes to check another claim, or goodbye to end.",
            language='en-IN'
        )
        response.redirect(f'/ivr/v2/continue?call_sid={call_sid}')
        return Response(str(response), mimetype='application/xml')

    except Exception as e:
        logger.error(f"Continue conversation error: {e}")
        return _error_response()


def _analyze_voice_authenticity(recording_url: str, session: Dict) -> float:
    """Voice deepfake detection using multimodal AI"""
    try:
        import requests

        audio_data = requests.get(recording_url, timeout=15).content

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            gemini_service = get_gemini_service()
            model = gemini_service.model

            import google.generativeai as genai
            audio_file = genai.upload_file(temp_path)

            prompt = """Analyze this audio for voice manipulation indicators:

1. Voice synthesis artifacts
2. Unnatural prosody patterns
3. Inconsistent background noise
4. Digital manipulation signatures
5. Splicing or concatenation

Rate manipulation risk from 0.0 (authentic) to 1.0 (likely fake).
Respond with just a number."""

            response = model.generate_content([prompt, audio_file])
            risk_text = response.text.strip()

            try:
                risk_score = float(risk_text)
                risk_score = max(0.0, min(1.0, risk_score))
            except:
                risk_score = 0.0

            logger.info(f"Voice authenticity check: risk={risk_score}")

            session['voice_deepfake_risk'] = risk_score

            return risk_score

        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.warning(f"Voice authenticity check failed: {e}")
        return 0.0


def _instant_fact_check(claim: str, language: str, session: Dict) -> Dict[str, Any]:
    """Instant fact-checking using Gemini 2.0 Flash"""
    try:
        start_time = time.time()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {
            'pib': [],
            'web': [],
            'gemini': {}
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_pib = executor.submit(search_pib_factcheck, claim, 2)
            future_web = executor.submit(search_web, claim, 3)
            future_gemini = executor.submit(_gemini_instant_analysis, claim, language)

            try:
                results['pib'] = future_pib.result(timeout=4)
            except:
                pass

            try:
                results['web'] = future_web.result(timeout=4)
            except:
                pass

            try:
                results['gemini'] = future_gemini.result(timeout=5)
            except Exception as e:
                logger.error(f"Gemini analysis failed: {e}")
                results['gemini'] = _fallback_analysis(claim)

        processing_time = time.time() - start_time
        logger.info(f"Instant fact-check completed in {processing_time:.2f}s")

        final_result = results['gemini']
        final_result['sources_found'] = len(results['pib']) + len(results['web'])
        final_result['processing_time'] = processing_time
        final_result['claim'] = claim

        _track_verification_analytics(claim, final_result, session)

        return final_result

    except Exception as e:
        logger.error(f"Instant fact-check error: {e}")
        return _fallback_analysis(claim)


def _gemini_instant_analysis(claim: str, language: str) -> Dict[str, Any]:
    """Use Gemini 2.0 Flash for instant analysis"""
    try:
        gemini_service = get_gemini_service()
        model = gemini_service.model

        prompt = f"""You are a fact-checking AI for SatyaCheck in India.

Analyze this claim INSTANTLY: "{claim}"

Provide:
1. VERDICT: True/False/Misleading/Partly True/Unverifiable
2. CONFIDENCE: 0-100
3. SHORT EXPLANATION (max 50 words, suitable for voice)
4. KEY REASON (one sentence)

Respond in JSON:
{{"verdict": "...", "confidence": 85, "explanation": "...", "reason": "..."}}"""

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 200,
            }
        )

        result_text = response.text.strip()
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0]

        result = json.loads(result_text)

        return {
            'verdict': result.get('verdict', 'Unverifiable'),
            'confidence': result.get('confidence', 50) / 100.0,
            'explanation': result.get('explanation', ''),
            'reason': result.get('reason', ''),
            'method': 'gemini_2.0_flash'
        }

    except Exception as e:
        logger.error(f"Gemini instant analysis error: {e}")
        raise


def _fallback_analysis(claim: str) -> Dict[str, Any]:
    """Fallback when all else fails"""
    return {
        'verdict': 'Unverifiable',
        'confidence': 0.3,
        'explanation': 'Unable to verify this claim instantly. Please try again or check our website.',
        'reason': 'Analysis timeout',
        'method': 'fallback'
    }


def _send_sms_summary(phone_number: str, result: Dict, claim: str):
    """Send SMS summary of fact-check result"""
    try:
        if not os.getenv('TWILIO_ACCOUNT_SID'):
            return

        client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )

        verdict = result.get('verdict', 'Unverifiable')
        confidence = int(result.get('confidence', 0) * 100)

        verdict_emoji = {
            'True': '',
            'False': '',
            'Misleading': '',
            'Partly True': '',
            'Unverifiable': ''
        }

        emoji = verdict_emoji.get(verdict, '')

        message = f"""SatyaCheck AI Result:

{emoji} Verdict: {verdict}
Confidence: {confidence}%

Claim: {claim[:100]}...

Powered by Google Gemini 2.0
Visit: satyacheck.ai"""

        client.messages.create(
            from_=os.getenv('TWILIO_PHONE_NUMBER'),
            to=phone_number,
            body=message
        )

        logger.info(f"SMS summary sent to {phone_number}")

    except Exception as e:
        logger.warning(f"SMS send failed: {e}")


def _track_call_analytics(caller_number: str, call_sid: str):
    """Track call analytics"""
    try:
        ivr_analytics['total_calls'] += 1

        hour = datetime.now().hour
        ivr_analytics['peak_hours'][hour] = ivr_analytics['peak_hours'].get(hour, 0) + 1

        storage = get_storage()
        storage.db.collection('ivr_analytics').document(call_sid).set({
            'caller': caller_number,
            'timestamp': datetime.now(),
            'call_sid': call_sid,
            'status': 'initiated'
        })

    except Exception as e:
        logger.warning(f"Analytics tracking failed: {e}")


def _track_verification_analytics(claim: str, result: Dict, session: Dict):
    """Track verification analytics"""
    try:
        verdict = result.get('verdict', 'Unverifiable')

        if verdict in ['False', 'Misleading']:
            ivr_analytics['fake_detected'] += 1

        ivr_analytics['claims_verified'] += 1

        lang = session.get('language', 'unknown')
        ivr_analytics['languages_used'][lang] = ivr_analytics['languages_used'].get(lang, 0) + 1

        logger.info(f"Analytics: Total calls={ivr_analytics['total_calls']}, "
                   f"Verified={ivr_analytics['claims_verified']}, "
                   f"Fake detected={ivr_analytics['fake_detected']}")

    except Exception as e:
        logger.warning(f"Verification analytics failed: {e}")


@ivr_advanced_bp.route('/ivr/v2/analytics', methods=['GET'])
def get_analytics():
    """Get IVR analytics dashboard data"""
    try:
        return {
            'success': True,
            'analytics': ivr_analytics,
            'timestamp': datetime.now().isoformat()
        }, 200
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500


def _transcribe_with_enhanced_google(audio_url: str, language: str) -> str:
    """Enhanced transcription with better accuracy"""
    try:
        import requests
        from google.cloud import speech

        audio_data = requests.get(audio_url, timeout=15).content

        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(content=audio_data)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            language_code='hi-IN' if language == 'hi' else 'en-IN',
            alternative_language_codes=['en-IN', 'hi-IN', 'ta-IN', 'te-IN'],
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
            model='phone_call',
            use_enhanced=True
        )

        response = client.recognize(config=config, audio=audio)

        if response.results:
            transcription = ' '.join([result.alternatives[0].transcript
                                     for result in response.results])
            logger.info(f"Enhanced transcription: {transcription}")
            return transcription

        return ""

    except Exception as e:
        logger.error(f"Enhanced transcription error: {e}")
        return ""


def _format_voice_result_v2(result: Dict, language: str, deepfake_risk: float) -> str:
    """Format result for voice delivery"""

    verdict = result.get('verdict', 'Unverifiable')
    confidence = int(result.get('confidence', 0) * 100)
    explanation = result.get('explanation', '')

    if language == 'hi':
        if verdict == 'False':
            intro = "यह गलत जानकारी है। "
        elif verdict == 'True':
            intro = "यह सही है। "
        elif verdict == 'Misleading':
            intro = "यह भ्रामक है। "
        else:
            intro = "इसकी पुष्टि नहीं हो सकी। "

        message = f"{intro}विश्वास स्तर {confidence} प्रतिशत। {explanation}"

        if deepfake_risk > 0.7:
            message += " चेतावनी: आवाज़ में हेरफेर की संभावना है।"
    else:
        if verdict == 'False':
            intro = "This is FALSE information. "
        elif verdict == 'True':
            intro = "This is TRUE. "
        elif verdict == 'Misleading':
            intro = "This is MISLEADING. "
        else:
            intro = "This cannot be verified. "

        message = f"{intro}Confidence level: {confidence} percent. {explanation}"

        if deepfake_risk > 0.7:
            message += " Warning: Possible voice manipulation detected."

    return message


def _end_call_with_summary(call_sid: str, session: Dict) -> Response:
    """End call with summary"""
    response = VoiceResponse()

    claims_count = session.get('claims_checked', 0)
    duration = int(time.time() - session.get('start_time', time.time()))

    summary = (
        f"Thank you for using Satya Check A I! "
        f"You verified {claims_count} claim{'s' if claims_count != 1 else ''} today. "
        f"Call duration: {duration} seconds. "
        f"Stay informed, stay safe. "
        f"Goodbye!"
    )

    response.say(summary, language='en-IN', voice='Polly.Aditi')
    response.hangup()

    _delete_session(call_sid)

    return Response(str(response), mimetype='application/xml')


def _get_tts_language(language: str) -> str:
    """Get TTS language code"""
    language_map = {
        'hi': 'hi-IN',
        'en': 'en-IN',
        'ta': 'ta-IN',
        'te': 'te-IN',
        'bn': 'bn-IN',
        'mr': 'mr-IN',
        'gu': 'gu-IN'
    }
    return language_map.get(language, 'en-IN')


def _error_response() -> Response:
    """Generic error response"""
    response = VoiceResponse()
    response.say(
        "We're sorry, an error occurred. Please try again later.",
        language='en-IN'
    )
    response.hangup()
    return Response(str(response), mimetype='application/xml')


def _save_session(call_sid: str, session: Dict):
    """Save session to Redis or memory"""
    try:
        if redis_available:
            redis_client.setex(
                f"ivr_session:{call_sid}",
                3600,
                json.dumps(session, default=str)
            )
        else:
            ivr_sessions[call_sid] = session
    except Exception as e:
        logger.warning(f"Session save failed: {e}")
        ivr_sessions[call_sid] = session


def _get_session(call_sid: str) -> Dict:
    """Get session from Redis or memory"""
    try:
        if redis_available:
            data = redis_client.get(f"ivr_session:{call_sid}")
            if data:
                return json.loads(data)

        return ivr_sessions.get(call_sid, {})
    except:
        return ivr_sessions.get(call_sid, {})


def _delete_session(call_sid: str):
    """Delete session"""
    try:
        if redis_available:
            redis_client.delete(f"ivr_session:{call_sid}")
        if call_sid in ivr_sessions:
            del ivr_sessions[call_sid]
    except:
        pass


logger.info("Advanced IVR system initialized")
