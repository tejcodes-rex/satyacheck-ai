"""
SatyaCheck AI - IVR System
Twilio-powered voice fact-checking with multi-language support
"""

import logging
import os
from typing import Dict, Optional
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client

from main import search_pib_factcheck
from ai_analyzer import analyze_claim_with_ai
from utils import generate_request_id

logger = logging.getLogger(__name__)

try:
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_IVR_NUMBER = '+12299222178'

    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info(f"Twilio SMS client initialized with IVR number: {TWILIO_IVR_NUMBER}")
    else:
        twilio_client = None
        logger.warning("Twilio credentials not found, SMS disabled")
except Exception as e:
    twilio_client = None
    logger.error(f"Failed to initialize Twilio client: {e}")

ivr_sessions = {}

LANGUAGE_MAP = {
    '1': {'code': 'hi-IN', 'short': 'hi', 'name': 'Hindi'},
    '2': {'code': 'en-IN', 'short': 'en', 'name': 'English'},
    '3': {'code': 'ta-IN', 'short': 'ta', 'name': 'Tamil'},
    '4': {'code': 'te-IN', 'short': 'te', 'name': 'Telugu'},
    '5': {'code': 'bn-IN', 'short': 'bn', 'name': 'Bengali'},
    '6': {'code': 'mr-IN', 'short': 'mr', 'name': 'Marathi'},
    '7': {'code': 'gu-IN', 'short': 'gu', 'name': 'Gujarati'}
}


def handle_incoming_call(request=None):
    """Handle incoming IVR call with language selection"""
    call_sid = None
    try:
        if request is None:
            from flask import request as flask_request
            request = flask_request

        from_number = request.form.get('From', 'Unknown')
        to_number = request.form.get('To', 'Unknown')
        call_direction = request.form.get('Direction', 'inbound')

        if call_direction == 'outbound-api':
            caller_number = to_number
        else:
            caller_number = from_number

        call_sid = request.form.get('CallSid', 'Unknown')

        if not call_sid or call_sid == 'Unknown':
            logger.error(f"[IVR] Missing CallSid in request")
            return _error_response()

        logger.info(f"[IVR] Incoming call from {caller_number} | CallSid: {call_sid}")

        try:
            ivr_sessions[call_sid] = {
                'caller': caller_number,
                'language': 'en-IN',
                'language_short': 'en',
                'state': 'language_selection',
                'retry_count': 0,
                'created_at': __import__('time').time()
            }
        except Exception as session_error:
            logger.error(f"[IVR] Failed to create session for {call_sid}: {session_error}")
            return _error_response()

        response = VoiceResponse()

        gather = Gather(
            num_digits=1,
            action='/ivr/language',
            method='POST',
            timeout=5
        )

        gather.say(
            '<speak>'
            '<prosody rate="medium" pitch="medium">'
            'Welcome to <break time="200ms"/> Satya Check AI. <break time="400ms"/> '
            'Please select your language. <break time="300ms"/> '
            'Press 1 for Hindi. <break time="200ms"/> '
            'Press 2 for English. <break time="200ms"/> '
            'Press 3 for Tamil. <break time="200ms"/> '
            'Press 4 for Telugu. <break time="200ms"/> '
            'Press 5 for Bengali. <break time="200ms"/> '
            'Press 6 for Marathi. <break time="200ms"/> '
            'Press 7 for Gujarati.'
            '</prosody>'
            '</speak>',
            voice='Google.en-IN-Neural2-A',
            language='en-IN'
        )

        response.append(gather)

        response.say(
            '<speak><prosody rate="medium">We didn\'t receive your selection. <break time="200ms"/> Please try again.</prosody></speak>',
            voice='Google.en-IN-Neural2-A',
            language='en-IN'
        )
        response.redirect('/ivr/voice')

        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)

    except Exception as e:
        logger.error(f"[IVR] Incoming call error for {call_sid}: {e}", exc_info=True)
        return _error_response()


def handle_language_selection(request=None):
    """Handle language selection and prompt for claim"""
    call_sid = None
    try:
        if request is None:
            from flask import request as flask_request
            request = flask_request

        call_sid = request.form.get('CallSid', 'Unknown')
        digits = request.form.get('Digits', '2')

        if not call_sid or call_sid == 'Unknown':
            logger.error(f"[IVR] Missing CallSid in language selection")
            return _error_response()

        if digits not in LANGUAGE_MAP:
            logger.warning(f"[IVR] Invalid language digit '{digits}' for {call_sid}, defaulting to English")
            digits = '2'

        language_info = LANGUAGE_MAP.get(digits, LANGUAGE_MAP['2'])
        language_code = language_info['code']
        language_short = language_info['short']

        try:
            if call_sid not in ivr_sessions:
                logger.warning(f"[IVR] Session not found for {call_sid}, creating new one")
                from_number = request.form.get('From', 'Unknown')
                to_number = request.form.get('To', 'Unknown')
                call_direction = request.form.get('Direction', 'inbound')
                caller_number = to_number if call_direction == 'outbound-api' else from_number

                ivr_sessions[call_sid] = {
                    'caller': caller_number,
                    'retry_count': 0
                }

            ivr_sessions[call_sid]['language'] = language_code
            ivr_sessions[call_sid]['language_short'] = language_short
            ivr_sessions[call_sid]['state'] = 'recording'

        except Exception as session_error:
            logger.error(f"[IVR] Session update error for {call_sid}: {session_error}")

        logger.info(f"[IVR] Language selected: {language_info['name']} for call {call_sid}")

        response = VoiceResponse()

        try:
            prompt_text = _get_prompt_text('speak_claim', language_short)
            voice = _get_voice_for_language(language_short)
            response.say(prompt_text, voice=voice, language=language_code)
        except Exception as prompt_error:
            logger.error(f"[IVR] Prompt error for {call_sid}: {prompt_error}")
            response.say("Please speak your claim after the beep.", voice='Google.en-IN-Neural2-A', language='en-IN')

        try:
            response.record(
                action=f'/ivr/analyze?lang={language_short}',
                method='POST',
                max_length=20,
                play_beep=True,
                timeout=7,
                finish_on_key='#',
                trim='trim-silence'
            )
        except Exception as record_error:
            logger.error(f"[IVR] Record setup error for {call_sid}: {record_error}")
            return _error_response()

        voice = _get_voice_for_language(language_short)
        response.say(
            _get_prompt_text('no_input', language_short),
            voice=voice,
            language=language_code
        )
        response.hangup()

        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)

    except Exception as e:
        logger.error(f"[IVR] Language selection error for {call_sid}: {e}", exc_info=True)
        return _error_response()


def handle_transcribe_callback(request=None):
    """Handle transcription callback from Twilio"""
    try:
        if request is None:
            from flask import request as flask_request
            request = flask_request

        transcription_text = request.form.get('TranscriptionText', '')
        recording_sid = request.form.get('RecordingSid', '')
        call_sid = request.args.get('call_sid', '')
        language = request.args.get('lang', 'en')

        logger.info(f"[Transcribe] ===== TRANSCRIPTION CALLBACK =====")
        logger.info(f"[Transcribe] CallSid: {call_sid}")
        logger.info(f"[Transcribe] RecordingSid: {recording_sid}")
        logger.info(f"[Transcribe] Language: {language}")
        logger.info(f"[Transcribe] Transcription: '{transcription_text}'")
        logger.info(f"[Transcribe] Length: {len(transcription_text)} characters")

        if not transcription_text or len(transcription_text.strip()) < 3:
            logger.error(f"[Transcribe] Empty or too short transcription for call {call_sid}")
            if call_sid not in ivr_sessions:
                ivr_sessions[call_sid] = {}
            ivr_sessions[call_sid]['result'] = {
                'verdict': 'Unverifiable',
                'confidence': 0.0,
                'explanation': 'Could not understand your speech. The recording may have been too short or unclear. Please try calling again and speak immediately after the beep.',
                'sources': []
            }
            ivr_sessions[call_sid]['transcription'] = 'Speech not recognized'
            logger.info(f"[Transcribe] Empty transcription result stored for {call_sid}")
            return ('', 200)

        if call_sid not in ivr_sessions:
            logger.warning(f"[Transcribe] Session not found for {call_sid}, creating new one")
            ivr_sessions[call_sid] = {}

        logger.info(f"[Transcribe] Starting analysis for: '{transcription_text}'")
        result = _analyze_claim(transcription_text, language)
        logger.info(f"[Transcribe] Analysis complete: {result.get('verdict', 'Unknown')}")

        ivr_sessions[call_sid]['result'] = result
        ivr_sessions[call_sid]['transcription'] = transcription_text

        logger.info(f"[Transcribe] Analysis stored for {call_sid}: {result.get('verdict', 'error')}")
        logger.info(f"[Transcribe] ===== TRANSCRIPTION CALLBACK COMPLETE =====")

        return ('', 200)

    except Exception as e:
        logger.error(f"[Transcribe] CRITICAL ERROR in transcription callback: {e}", exc_info=True)
        if call_sid and call_sid in ivr_sessions:
            ivr_sessions[call_sid]['result'] = {
                'verdict': 'Error',
                'confidence': 0.0,
                'explanation': 'System error occurred during transcription. Please try again.',
                'sources': []
            }
        return ('', 500)


def handle_analyze(request=None):
    """Handle recording analysis using Google Speech-to-Text"""
    try:
        if request is None:
            from flask import request as flask_request
            request = flask_request

        call_sid = request.form.get('CallSid', '')
        recording_url = request.form.get('RecordingUrl', '')
        language = request.args.get('lang', 'en')
        language_code = f'{language}-IN' if language != 'en' else 'en-IN'

        logger.info(f"[IVR] Analyze handler: CallSid={call_sid}, RecordingUrl={recording_url}, Lang={language}")

        response = VoiceResponse()
        voice = _get_voice_for_language(language)

        if not recording_url:
            logger.error(f"[IVR] No recording URL for {call_sid}")
            response.say(
                '<speak><prosody rate="medium">Recording not received. <break time="300ms"/> Please try again. <break time="200ms"/> Goodbye.</prosody></speak>',
                voice=voice,
                language=language_code
            )
            response.hangup()
            headers = {'Content-Type': 'application/xml'}
            return (str(response), 200, headers)

        response.say(
            '<speak><prosody rate="medium">Got it! <break time="300ms"/> Now analyzing your claim. <break time="200ms"/> This will take just a few seconds.</prosody></speak>',
            voice=voice,
            language=language_code
        )

        import threading
        def process_recording():
            try:
                logger.info(f"[IVR] Background: Downloading recording for {call_sid}")

                transcription = _transcribe_recording_with_google(recording_url, language)

                if not transcription or len(transcription.strip()) < 3:
                    logger.error(f"[IVR] Background: Empty transcription for {call_sid}")
                    if call_sid not in ivr_sessions:
                        ivr_sessions[call_sid] = {}
                    ivr_sessions[call_sid]['result'] = {
                        'verdict': 'Unverifiable',
                        'confidence': 0.0,
                        'explanation': 'Could not understand your speech. Please speak clearly and try again.',
                        'sources': []
                    }
                    ivr_sessions[call_sid]['transcription'] = 'Speech not recognized'
                    return

                logger.info(f"[IVR] Background: Transcribed '{transcription[:50]}...' for {call_sid}")

                result = _analyze_claim(transcription, language)

                if call_sid not in ivr_sessions:
                    ivr_sessions[call_sid] = {}
                ivr_sessions[call_sid]['result'] = result
                ivr_sessions[call_sid]['transcription'] = transcription

                logger.info(f"[IVR] Background: Analysis complete for {call_sid}: {result.get('verdict', 'N/A')}")

            except Exception as bg_error:
                logger.error(f"[IVR] Background processing error for {call_sid}: {bg_error}", exc_info=True)
                if call_sid not in ivr_sessions:
                    ivr_sessions[call_sid] = {}
                ivr_sessions[call_sid]['result'] = {
                    'verdict': 'Error',
                    'confidence': 0.0,
                    'explanation': 'System error occurred. Please try again.',
                    'sources': []
                }

        thread = threading.Thread(target=process_recording)
        thread.daemon = True
        thread.start()

        response.pause(length=6)

        response.redirect(f'/ivr/results?lang={language}')

        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)

    except Exception as e:
        logger.error(f"[IVR] Analyze error: {e}", exc_info=True)
        return _error_response()


def handle_results(request=None):
    """Retrieve and speak fact-check results with SMS delivery"""
    call_sid = None
    try:
        if request is None:
            from flask import request as flask_request
            request = flask_request

        call_sid = request.form.get('CallSid', 'Unknown')
        language = request.args.get('lang', 'en')
        language_code = f'{language}-IN' if language != 'en' else 'en-IN'

        if not call_sid or call_sid == 'Unknown':
            logger.error(f"[IVR] Missing CallSid in results handler")
            return _error_response()

        logger.info(f"[IVR] Results handler called for {call_sid}")

        response = VoiceResponse()

        session = ivr_sessions.get(call_sid, {})
        result = session.get('result')

        caller_number = session.get('caller', 'Unknown')
        if caller_number == 'Unknown':
            from_number = request.form.get('From', 'Unknown')
            to_number = request.form.get('To', 'Unknown')
            call_direction = request.form.get('Direction', 'inbound')

            if call_direction == 'outbound-api':
                caller_number = to_number
            else:
                caller_number = from_number

            logger.info(f"[IVR] Caller number from request: {caller_number} (direction: {call_direction})")

        transcription = session.get('transcription', 'Unknown claim')

        voice = _get_voice_for_language(language)

        if not result:
            retry_count = session.get('retry_count', 0)

            if retry_count < 3:
                logger.info(f"[IVR] Result not ready for {call_sid}, retrying ({retry_count}/3)...")

                try:
                    if call_sid in ivr_sessions:
                        ivr_sessions[call_sid]['retry_count'] = retry_count + 1
                except Exception as retry_error:
                    logger.error(f"[IVR] Error updating retry count: {retry_error}")

                response.say(
                    '<speak><prosody rate="medium">Almost there! <break time="300ms"/> Just a moment.</prosody></speak>',
                    voice=voice,
                    language=language_code
                )
                response.pause(length=5)
                response.redirect(f'/ivr/results?lang={language}')
            else:
                logger.error(f"[IVR] Result timeout for {call_sid}")
                response.say(
                    '<speak><prosody rate="medium">Sorry, <break time="200ms"/> processing is taking longer than expected. <break time="300ms"/> Please try calling again. <break time="200ms"/> Goodbye.</prosody></speak>',
                    voice=voice,
                    language=language_code
                )
                response.hangup()

                _cleanup_session(call_sid, delay_seconds=60)
        else:
            try:
                speech_text = _format_result_for_speech(result, language)
                logger.info(f"[IVR] Speaking result for {call_sid}: {result.get('verdict', 'N/A')}")

                voice = _get_voice_for_language(language)
                response.say(
                    speech_text,
                    voice=voice,
                    language=language_code
                )
            except Exception as speech_error:
                logger.error(f"[IVR] Error formatting/speaking result: {speech_error}")
                response.say(
                    f"Your claim is {result.get('verdict', 'unverifiable')}.",
                    voice=voice,
                    language=language_code
                )

            try:
                import threading
                sms_thread = threading.Thread(
                    target=_send_detailed_sms,
                    args=(caller_number, transcription, result, language)
                )
                sms_thread.daemon = True
                sms_thread.start()
                logger.info(f"[IVR] SMS sending initiated for {caller_number}")

                response.say(
                    '<speak><prosody rate="medium"><break time="300ms"/> A detailed report has been sent to your phone via text message.</prosody></speak>',
                    voice=voice,
                    language=language_code
                )
            except Exception as sms_error:
                logger.error(f"[IVR] Failed to initiate SMS: {sms_error}")

            try:
                response.say(
                    _get_prompt_text('goodbye', language),
                    voice=voice,
                    language=language_code
                )
            except Exception as goodbye_error:
                logger.error(f"[IVR] Error with goodbye message: {goodbye_error}")
                response.say("Goodbye!", voice=voice, language=language_code)

            response.hangup()

            _cleanup_session(call_sid, delay_seconds=60)

        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)

    except Exception as e:
        logger.error(f"[IVR] Results error for {call_sid}: {e}", exc_info=True)
        if call_sid and call_sid != 'Unknown':
            _cleanup_session(call_sid, delay_seconds=30)
        return _error_response()


def _process_recording_sync(call_sid: str, recording_url: str, language: str):
    """Synchronous recording processing with Google Speech"""
    try:
        import requests
        import time
        start_time = time.time()

        logger.info(f"[{call_sid}] Starting sync processing...")

        # Initialize session result if not exists
        if call_sid not in ivr_sessions:
            ivr_sessions[call_sid] = {}

        if not recording_url.endswith('.wav'):
            recording_url = recording_url + '.wav'

        logger.info(f"[{call_sid}] Downloading recording from: {recording_url}")

        time.sleep(1)

        response = requests.get(
            recording_url,
            auth=(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN')),
            timeout=15
        )

        if response.status_code != 200:
            logger.error(f"[{call_sid}] Failed to download recording: {response.status_code}")
            ivr_sessions[call_sid]['result'] = {'error': 'Could not download recording. Please try again.'}
            return

        audio_data = response.content
        logger.info(f"[{call_sid}] Downloaded {len(audio_data)} bytes")

        if len(audio_data) < 1000:
            logger.error(f"[{call_sid}] Recording too small: {len(audio_data)} bytes")
            ivr_sessions[call_sid]['result'] = {'error': 'Recording was too short. Please speak clearly.'}
            return

        logger.info(f"[{call_sid}] Transcribing with Google Speech...")
        try:
            from content_processor import get_content_processor
            processor = get_content_processor()

            language_map = {
                'hi': 'hi-IN',
                'en': 'en-US',
                'ta': 'ta-IN',
                'te': 'te-IN',
                'bn': 'bn-IN',
                'mr': 'mr-IN',
                'gu': 'gu-IN'
            }
            speech_language = language_map.get(language, 'en-US')

            result = processor.process_content(audio_data, 'audio', {'language': speech_language})

            if not result.get('success') or not result.get('extracted_text'):
                logger.error(f"[{call_sid}] Transcription failed: {result}")
                ivr_sessions[call_sid]['result'] = {'error': 'Could not understand speech. Please speak clearly.'}
                return

            transcription = result.get('extracted_text', '').strip()
            logger.info(f"[{call_sid}] Transcribed: '{transcription}'")

        except Exception as transcribe_error:
            logger.error(f"[{call_sid}] Transcription error: {transcribe_error}", exc_info=True)
            ivr_sessions[call_sid]['result'] = {'error': 'Transcription service unavailable. Please try again.'}
            return

        logger.info(f"[{call_sid}] Analyzing claim...")
        analysis_result = _analyze_claim(transcription, language)

        elapsed = time.time() - start_time
        logger.info(f"[{call_sid}] Analysis complete in {elapsed:.1f}s: {analysis_result.get('verdict', 'N/A')}")

        if call_sid in ivr_sessions:
            ivr_sessions[call_sid]['result'] = analysis_result
            ivr_sessions[call_sid]['transcription'] = transcription

    except Exception as e:
        logger.error(f"[{call_sid}] Sync processing error: {e}", exc_info=True)
        if call_sid in ivr_sessions:
            ivr_sessions[call_sid]['result'] = {'error': 'Processing failed. Please try again.'}


def _transcribe_recording_with_google(recording_url: str, language: str) -> str:
    """Download Twilio recording and transcribe with Google Speech-to-Text"""
    import requests
    import io
    from google.cloud import speech_v1

    try:
        import time
        time.sleep(2)

        logger.info(f"[GoogleSpeech] Downloading recording from {recording_url[:50]}...")

        audio_url = recording_url + '.wav' if not recording_url.endswith('.wav') else recording_url

        import os
        auth = (os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                logger.info(f"[GoogleSpeech] Download attempt {attempt + 1}/{max_retries}...")
                response = requests.get(audio_url, auth=auth, timeout=15)
                response.raise_for_status()
                logger.info(f"[GoogleSpeech] Download successful!")
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404 and attempt < max_retries - 1:
                    logger.warning(f"[GoogleSpeech] Recording not ready yet (404), waiting 3 seconds...")
                    time.sleep(3)
                else:
                    logger.error(f"[GoogleSpeech] Download failed after {attempt + 1} attempts")
                    raise

        audio_content = response.content
        logger.info(f"[GoogleSpeech] Downloaded {len(audio_content)} bytes")

        client = speech_v1.SpeechClient()

        google_language_code = {
            'en': 'en-IN',
            'hi': 'hi-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'bn': 'bn-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN'
        }.get(language, 'en-IN')

        config = speech_v1.RecognitionConfig(
            language_code=google_language_code,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            alternative_language_codes=['en-US', 'hi-IN'] if google_language_code == 'en-IN' else []
        )

        audio = speech_v1.RecognitionAudio(content=audio_content)

        logger.info(f"[GoogleSpeech] Transcribing with language={google_language_code}, audio_size={len(audio_content)} bytes")
        logger.info(f"[GoogleSpeech] Config: auto-detect encoding/sample_rate, enhanced=True, alternative_languages={config.alternative_language_codes}")
        speech_response = client.recognize(config=config, audio=audio)

        transcription = ""
        logger.info(f"[GoogleSpeech] Speech response has {len(speech_response.results)} results")

        for i, result in enumerate(speech_response.results):
            logger.info(f"[GoogleSpeech] Result {i}: {len(result.alternatives)} alternatives")
            if result.alternatives:
                alternative = result.alternatives[0]
                transcription += alternative.transcript
                logger.info(f"[GoogleSpeech] Alternative 0: '{alternative.transcript}' (confidence: {alternative.confidence:.2%})")
            else:
                logger.warning(f"[GoogleSpeech] Result {i} has no alternatives")

        if transcription:
            logger.info(f"[GoogleSpeech] SUCCESS! Final transcription: '{transcription}'")
        else:
            logger.warning(f"[GoogleSpeech] Empty transcription - no speech detected in audio")

        return transcription.strip()

    except Exception as e:
        logger.error(f"[GoogleSpeech] Transcription error: {e}", exc_info=True)
        logger.error(f"[GoogleSpeech] Recording URL was: {recording_url}")
        return ""


def _analyze_claim(claim: str, language: str) -> Dict[str, any]:
    """Analyze claim using existing fact-checking pipeline"""
    try:
        logger.info(f"[Analysis] Starting analysis for claim: '{claim[:100]}'")

        try:
            pib_results = search_pib_factcheck(claim)
            logger.info(f"[Analysis] PIB search returned {len(pib_results) if pib_results else 0} results")
        except Exception as pib_error:
            logger.error(f"[Analysis] PIB search failed: {pib_error}")
            pib_results = []

        try:
            ai_result = analyze_claim_with_ai(
                claim=claim,
                pib_results=pib_results,
                target_language=language
            )
            logger.info(f"[Analysis] AI analysis returned: {ai_result.get('verdict', 'Unknown')}")
        except Exception as ai_error:
            logger.error(f"[Analysis] AI analysis failed: {ai_error}", exc_info=True)
            return {
                'verdict': 'Unverifiable',
                'confidence': 0.5,
                'explanation': 'Unable to verify this claim at this time. Please check trusted fact-checking websites.',
                'sources': []
            }

        if ai_result.get('verdict') == 'Error' or 'error' in ai_result:
            logger.error(f"[Analysis] AI returned error result: {ai_result.get('error', 'Unknown error')}")
            return {
                'verdict': 'Unverifiable',
                'confidence': 0.5,
                'explanation': 'Unable to complete analysis. Please try again or verify manually.',
                'sources': []
            }

        if ai_result.get('verdict') == 'Unverified' and 'technical limitations' in ai_result.get('explanation', '').lower():
            logger.warning(f"[Analysis] Got fallback unverified result - AI may have failed")
            return {
                'verdict': 'Unverifiable',
                'confidence': 0.5,
                'explanation': 'This claim could not be verified at this time. Please check trusted sources.',
                'sources': []
            }

        confidence = ai_result.get('score', 50)
        if confidence > 1:
            confidence = confidence / 100.0

        return {
            'verdict': ai_result.get('verdict', 'Unverifiable'),
            'confidence': confidence,
            'explanation': ai_result.get('explanation', 'Analysis completed.'),
            'sources': ai_result.get('sources', [])[:2]
        }

    except Exception as e:
        logger.error(f"[Analysis] Critical error in _analyze_claim: {e}", exc_info=True)
        return {
            'verdict': 'Unverifiable',
            'confidence': 0.5,
            'explanation': 'System error occurred. Please verify this claim manually.',
            'sources': []
        }


def _format_result_for_speech(result: Dict[str, any], language: str) -> str:
    """Format result for text-to-speech using SSML"""
    if result.get('error'):
        return f'<speak><prosody rate="medium">{result["error"]}</prosody></speak>'

    verdict = result.get('verdict', 'Unverifiable')
    confidence = result.get('confidence', 0)
    explanation = result.get('explanation', '')
    sources = result.get('sources', [])

    speech = '<speak><prosody rate="medium" pitch="medium">'

    speech += f'<emphasis level="strong">Your claim is {verdict}.</emphasis> '
    speech += '<break time="400ms"/> '

    speech += f'Our confidence level is <prosody rate="slow">{int(confidence * 100)} percent</prosody>. '
    speech += '<break time="400ms"/> '

    if explanation:
        explanation_short = explanation[:150].strip()
        if len(explanation) > 150:
            last_period = explanation_short.rfind('.')
            if last_period > 50:
                explanation_short = explanation_short[:last_period + 1]
            else:
                explanation_short += "..."

        speech += f'Here\'s why: <break time="300ms"/> {explanation_short} '
        speech += '<break time="400ms"/> '

    if sources:
        source_count = len(sources)
        speech += f'This verdict is based on {source_count} verified source{"s" if source_count > 1 else ""}. '
        speech += '<break time="300ms"/> '

    speech += '</prosody></speak>'
    return speech


def _get_voice_for_language(language: str) -> str:
    """Get Google Neural2 voice for language"""
    voices = {
        'hi': 'Google.hi-IN-Neural2-A',
        'en': 'Google.en-IN-Neural2-A',
        'ta': 'Google.ta-IN-Standard-A',
        'te': 'Google.te-IN-Standard-A',
        'bn': 'Google.bn-IN-Standard-A',
        'mr': 'Google.mr-IN-Standard-A',
        'gu': 'Google.gu-IN-Standard-A'
    }
    return voices.get(language, 'Google.en-IN-Neural2-A')


def _get_prompt_text(key: str, language: str) -> str:
    """Get localized prompts with SSML"""
    prompts = {
        'speak_claim': {
            'en': '<speak><prosody rate="slow">Perfect! <break time="500ms"/> You will hear a beep. <break time="300ms"/> Start speaking immediately after the beep. <break time="500ms"/> Speak your claim clearly. <break time="500ms"/> Press hash when done. <break time="300ms"/> Ready?</prosody></speak>',
            'hi': '<speak><prosody rate="slow">बहुत अच्छा! <break time="500ms"/> आपको एक बीप सुनाई देगी। <break time="300ms"/> बीप के तुरंत बाद बोलना शुरू करें। <break time="500ms"/> अपना दावा स्पष्ट रूप से बोलें। <break time="500ms"/> पूरा होने पर हैश दबाएं। <break time="300ms"/> तैयार?</prosody></speak>'
        },
        'no_input': {
            'en': '<speak><prosody rate="medium">We did not receive your recording. <break time="300ms"/> Please try calling again. <break time="200ms"/> Goodbye.</prosody></speak>',
            'hi': '<speak><prosody rate="medium">हमें आपकी रिकॉर्डिंग प्राप्त नहीं हुई। <break time="300ms"/> कृपया फिर से कॉल करें। <break time="200ms"/> नमस्ते।</prosody></speak>'
        },
        'analyzing': {
            'en': '<speak><prosody rate="medium">Perfect! Recording received. <break time="300ms"/> Now analyzing your claim using AI and verified sources. <break time="200ms"/> Please wait.</prosody></speak>',
            'hi': '<speak><prosody rate="medium">बढ़िया! रिकॉर्डिंग प्राप्त हुई। <break time="300ms"/> अब आपके दावे का AI और सत्यापित स्रोतों से विश्लेषण हो रहा है। <break time="200ms"/> कृपया प्रतीक्षा करें।</prosody></speak>'
        },
        'still_processing': {
            'en': '<speak><prosody rate="medium">Still processing your claim. <break time="300ms"/> Please wait a few more seconds.</prosody></speak>',
            'hi': '<speak><prosody rate="medium">अभी भी आपके दावे की प्रोसेसिंग हो रही है। <break time="300ms"/> कुछ और सेकंड प्रतीक्षा करें।</prosody></speak>'
        },
        'goodbye': {
            'en': '<speak><prosody rate="medium">Thank you for using Satya Check AI. <break time="300ms"/> Goodbye!</prosody></speak>',
            'hi': '<speak><prosody rate="medium">सत्य चेक AI का उपयोग करने के लिए धन्यवाद। <break time="300ms"/> नमस्ते!</prosody></speak>'
        }
    }

    return prompts.get(key, {}).get(language, prompts.get(key, {}).get('en', ''))


def _send_detailed_sms(caller_number: str, claim: str, result: Dict[str, any], language: str):
    """Send detailed analysis via SMS after voice call"""
    try:
        logger.info(f"[SMS] === Starting SMS send process ===")
        logger.info(f"[SMS] Caller number: {caller_number}")
        logger.info(f"[SMS] Claim: {claim[:50]}...")
        logger.info(f"[SMS] Result: {result.get('verdict', 'Unknown')}")

        if not twilio_client:
            logger.error(f"[SMS] Twilio client not initialized! SMS cannot be sent.")
            logger.error(f"[SMS] TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID[:10] if TWILIO_ACCOUNT_SID else 'NOT SET'}")
            logger.error(f"[SMS] TWILIO_AUTH_TOKEN: {'SET' if TWILIO_AUTH_TOKEN else 'NOT SET'}")
            return False

        if caller_number and caller_number.startswith('client:'):
            logger.info(f"[SMS] Skipping SMS for test client: {caller_number}")
            return False

        if not caller_number or caller_number == 'Unknown':
            logger.error(f"[SMS] Invalid caller number: {caller_number}")
            return False

        verdict = result.get('verdict', 'Unverifiable')
        confidence = result.get('confidence', 0.5)
        if isinstance(confidence, (int, float)):
            if confidence <= 1:
                confidence_pct = int(confidence * 100)
            else:
                confidence_pct = int(confidence)
        else:
            confidence_pct = 50

        explanation = result.get('explanation', 'No explanation available')
        sources = result.get('sources', [])

        sms_text = f"SatyaCheck Result:\n"
        sms_text += f"Claim: {claim[:60]}...\n"
        sms_text += f"Verdict: {verdict}\n"
        sms_text += f"Confidence: {confidence_pct}%\n"
        sms_text += f"Details: https://satyacheck.ai"

        logger.info(f"[SMS] Attempting to send SMS...")
        logger.info(f"[SMS] From: {TWILIO_IVR_NUMBER}")
        logger.info(f"[SMS] To: {caller_number}")
        logger.info(f"[SMS] Message length: {len(sms_text)} characters")

        message = twilio_client.messages.create(
            body=sms_text,
            from_=TWILIO_IVR_NUMBER,
            to=caller_number
        )

        logger.info(f"[SMS] SMS SENT SUCCESSFULLY!")
        logger.info(f"[SMS] Message SID: {message.sid}")
        logger.info(f"[SMS] Status: {message.status}")
        logger.info(f"[SMS] === SMS send complete ===")
        return True

    except Exception as e:
        logger.error(f"[SMS] CRITICAL ERROR sending SMS to {caller_number}: {e}", exc_info=True)
        logger.error(f"[SMS] Error type: {type(e).__name__}")
        logger.error(f"[SMS] Error details: {str(e)}")
        return False


def _cleanup_session(call_sid: str, delay_seconds: int = 300):
    """Clean up session after delay to prevent memory leaks"""
    try:
        import threading
        import time

        def cleanup_task():
            try:
                time.sleep(delay_seconds)
                if call_sid in ivr_sessions:
                    del ivr_sessions[call_sid]
                    logger.info(f"[Cleanup] Session {call_sid} cleaned up after {delay_seconds}s")
            except Exception as e:
                logger.error(f"[Cleanup] Error cleaning session {call_sid}: {e}")

        thread = threading.Thread(target=cleanup_task)
        thread.daemon = True
        thread.start()

    except Exception as e:
        logger.error(f"[Cleanup] Failed to schedule cleanup for {call_sid}: {e}")


def _error_response():
    """Generate error response"""
    try:
        response = VoiceResponse()
        response.say(
            '<speak><prosody rate="medium">We\'re sorry, <break time="200ms"/> an error occurred. <break time="300ms"/> Please try again later. <break time="200ms"/> Goodbye.</prosody></speak>',
            voice='Google.en-IN-Neural2-A',
            language='en-IN'
        )
        response.hangup()
        headers = {'Content-Type': 'application/xml'}
        return (str(response), 200, headers)
    except Exception as e:
        logger.error(f"[Error] Failed to generate error response: {e}")
        return ("Error occurred", 500, {'Content-Type': 'text/plain'})


logger.info("IVR bot module initialized with SMS support")
