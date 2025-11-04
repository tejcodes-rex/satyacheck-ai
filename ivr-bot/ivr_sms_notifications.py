"""
SatyaCheck AI - SMS Notification System
SMS delivery for IVR fact-checking results
"""

import logging
import os
from twilio.rest import Client
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from database import get_storage

logger = logging.getLogger(__name__)

try:
    twilio_client = Client(
        os.getenv('TWILIO_ACCOUNT_SID'),
        os.getenv('TWILIO_AUTH_TOKEN')
    )
    TWILIO_PHONE = os.getenv('TWILIO_PHONE_NUMBER')
    SMS_ENABLED = True
    logger.info("SMS notifications enabled")
except:
    twilio_client = None
    SMS_ENABLED = False
    logger.warning("SMS notifications disabled (Twilio not configured)")


def send_instant_result_sms(phone_number: str, claim: str, result: Dict,
                            language: str = 'en') -> bool:
    """Send instant SMS with fact-check result"""
    if not SMS_ENABLED:
        logger.warning("SMS not enabled")
        return False

    try:
        message = _format_result_sms(claim, result, language)

        sms = twilio_client.messages.create(
            from_=TWILIO_PHONE,
            to=phone_number,
            body=message
        )

        logger.info(f"SMS sent to {phone_number}: {sms.sid}")

        _log_sms_delivery(phone_number, claim, result, sms.sid)

        return True

    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        return False


def _format_result_sms(claim: str, result: Dict, language: str) -> str:
    """Format fact-check result for SMS"""

    verdict = result.get('verdict', 'Unverifiable')
    confidence = int(result.get('confidence', 0) * 100)
    explanation = result.get('explanation', '')[:150]

    if language == 'hi':
        message = f"""सत्य चेक AI परिणाम

निर्णय: {_translate_verdict(verdict, 'hi')}
विश्वास: {confidence}%

दावा: {claim[:100]}...

स्पष्टीकरण: {explanation}

Powered by Google Gemini 2.0
satyacheck.ai"""

    else:
        message = f"""SatyaCheck AI Result

Verdict: {verdict}
Confidence: {confidence}%

Claim: {claim[:100]}...

Explanation: {explanation}

Powered by Google Gemini 2.0
satyacheck.ai"""

    return message


def _translate_verdict(verdict: str, language: str) -> str:
    """Translate verdict to local language"""
    translations = {
        'hi': {
            'True': 'सत्य',
            'False': 'झूठ',
            'Misleading': 'भ्रामक',
            'Partly True': 'आंशिक सत्य',
            'Unverifiable': 'अप्रमाणित',
            'Satire': 'व्यंग्य'
        }
    }

    return translations.get(language, {}).get(verdict, verdict)


def send_daily_digest_sms(phone_number: str, language: str = 'en') -> bool:
    """Send daily digest of all claims checked by user"""
    if not SMS_ENABLED:
        return False

    try:
        storage = get_storage()
        db = storage.db
        yesterday = datetime.now() - timedelta(days=1)

        claims_ref = db.collection('ivr_analytics').where(
            'caller', '==', phone_number
        ).where(
            'timestamp', '>=', yesterday
        ).stream()

        claims_count = 0
        fake_count = 0
        true_count = 0

        for claim_doc in claims_ref:
            claims_count += 1
            data = claim_doc.to_dict()
            verdict = data.get('verdict', '')

            if verdict in ['False', 'Misleading']:
                fake_count += 1
            elif verdict == 'True':
                true_count += 1

        if claims_count == 0:
            return False

        if language == 'hi':
            message = f"""सत्य चेक AI - दैनिक सारांश

आज आपने {claims_count} दावों की जांच की:
सत्य: {true_count}
झूठ: {fake_count}
अन्य: {claims_count - fake_count - true_count}

आप सूचित रहे!

कल फिर मिलेंगे।
satyacheck.ai"""
        else:
            message = f"""SatyaCheck AI - Daily Digest

Today you checked {claims_count} claims:
True: {true_count}
False: {fake_count}
Other: {claims_count - fake_count - true_count}

Stay informed!

See you tomorrow!
satyacheck.ai"""

        sms = twilio_client.messages.create(
            from_=TWILIO_PHONE,
            to=phone_number,
            body=message
        )

        logger.info(f"Daily digest sent to {phone_number}")
        return True

    except Exception as e:
        logger.error(f"Daily digest failed: {e}")
        return False


def send_emergency_alert(phone_numbers: List[str], alert: Dict,
                        language: str = 'en') -> int:
    """Send emergency misinformation alert to multiple users"""
    if not SMS_ENABLED:
        return 0

    title = alert.get('title', '')
    description = alert.get('description', '')
    urgency = alert.get('urgency', 'medium')

    urgency_emoji = {
        'low': '',
        'medium': '',
        'high': '',
        'critical': ''
    }

    emoji = urgency_emoji.get(urgency, '')

    success_count = 0

    for phone_number in phone_numbers:
        try:
            if language == 'hi':
                message = f"""{emoji} आपातकालीन चेतावनी

{title}

{description[:200]}

यह संदेश देखते ही साझा करें।

सत्य चेक AI
satyacheck.ai"""
            else:
                message = f"""{emoji} EMERGENCY ALERT

{title}

{description[:200]}

Please share immediately.

SatyaCheck AI
satyacheck.ai"""

            twilio_client.messages.create(
                from_=TWILIO_PHONE,
                to=phone_number,
                body=message
            )

            success_count += 1
            logger.info(f"Emergency alert sent to {phone_number}")

        except Exception as e:
            logger.error(f"Emergency alert failed for {phone_number}: {e}")
            continue

    logger.info(f"Emergency alert sent to {success_count}/{len(phone_numbers)} users")
    return success_count


def send_detailed_report_sms(phone_number: str, claim_id: str,
                             language: str = 'en') -> bool:
    """Send SMS with link to detailed web report"""
    if not SMS_ENABLED:
        return False

    try:
        report_url = f"https://satyacheck.ai/report/{claim_id}"

        if language == 'hi':
            message = f"""विस्तृत रिपोर्ट तैयार है!

आपके दावे की पूरी रिपोर्ट यहाँ देखें:
{report_url}

रिपोर्ट में:
सभी स्रोत
AI विश्लेषण
संदर्भ जानकारी
साझा करने योग्य लिंक

सत्य चेक AI"""
        else:
            message = f"""Detailed Report Ready!

View your complete fact-check report:
{report_url}

Report includes:
All sources
AI analysis
Context information
Shareable link

SatyaCheck AI"""

        twilio_client.messages.create(
            from_=TWILIO_PHONE,
            to=phone_number,
            body=message
        )

        logger.info(f"Report link sent to {phone_number}")
        return True

    except Exception as e:
        logger.error(f"Report link SMS failed: {e}")
        return False


def send_missed_call_followup(phone_number: str, language: str = 'en') -> bool:
    """Send SMS follow-up for missed/dropped calls"""
    if not SMS_ENABLED:
        return False

    try:
        if language == 'hi':
            message = """सत्य चेक AI

हमें खेद है कि आपकी कॉल पूरी नहीं हो सकी।

फिर से कोशिश करें:
[PHONE_NUMBER]

या WhatsApp भेजें:
[WHATSAPP_NUMBER]

हम 24/7 उपलब्ध हैं।"""
        else:
            message = """SatyaCheck AI

Sorry your call was interrupted.

Try again:
[PHONE_NUMBER]

Or WhatsApp us:
[WHATSAPP_NUMBER]

We're available 24/7!"""

        twilio_client.messages.create(
            from_=TWILIO_PHONE,
            to=phone_number,
            body=message
        )

        logger.info(f"Missed call follow-up sent to {phone_number}")
        return True

    except Exception as e:
        logger.error(f"Missed call SMS failed: {e}")
        return False


def _log_sms_delivery(phone_number: str, claim: str, result: Dict, sms_sid: str):
    """Log SMS delivery for analytics"""
    try:
        storage = get_storage()
        db = storage.db
        db.collection('sms_logs').document(sms_sid).set({
            'phone_number': phone_number,
            'claim': claim[:100],
            'verdict': result.get('verdict'),
            'confidence': result.get('confidence'),
            'timestamp': datetime.now(),
            'sms_sid': sms_sid,
            'status': 'sent'
        })
    except Exception as e:
        logger.warning(f"SMS logging failed: {e}")


def get_sms_analytics() -> Dict[str, Any]:
    """Get SMS delivery analytics"""
    try:
        storage = get_storage()
        db = storage.db

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        sms_ref = db.collection('sms_logs').where(
            'timestamp', '>=', today
        ).stream()

        total_sent = 0
        by_verdict = {}

        for sms_doc in sms_ref:
            total_sent += 1
            data = sms_doc.to_dict()
            verdict = data.get('verdict', 'Unknown')
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

        return {
            'total_sent_today': total_sent,
            'by_verdict': by_verdict,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"SMS analytics failed: {e}")
        return {}


def subscribe_to_alerts(phone_number: str, alert_types: List[str] = None) -> bool:
    """Subscribe user to SMS alerts"""
    try:
        if alert_types is None:
            alert_types = ['daily_digest', 'emergency_alerts']

        storage = get_storage()
        db = storage.db
        db.collection('sms_subscriptions').document(phone_number).set({
            'phone_number': phone_number,
            'subscribed_at': datetime.now(),
            'alert_types': alert_types,
            'active': True
        })

        logger.info(f"{phone_number} subscribed to SMS alerts")
        return True

    except Exception as e:
        logger.error(f"Subscription failed: {e}")
        return False


def unsubscribe_from_alerts(phone_number: str) -> bool:
    """Unsubscribe user from SMS alerts"""
    try:
        storage = get_storage()
        db = storage.db
        db.collection('sms_subscriptions').document(phone_number).update({
            'active': False,
            'unsubscribed_at': datetime.now()
        })

        logger.info(f"{phone_number} unsubscribed from SMS alerts")
        return True

    except Exception as e:
        logger.error(f"Unsubscribe failed: {e}")
        return False


logger.info("SMS Notifications System Initialized")
