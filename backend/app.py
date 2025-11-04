"""
SatyaCheck AI - Flask Application Wrapper
Production-ready app for Cloud Run with Gunicorn
"""

import os
import logging
import signal
import time
from datetime import datetime
from functools import wraps
from flask import Flask, request, send_file, g
from flask_cors import CORS
from main import satyacheck_main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "X-Response-Time"],
        "max_age": 3600
    }
})

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))
service_health = {
    'start_time': datetime.utcnow(),
    'requests_processed': 0,
    'requests_failed': 0,
    'last_request': None
}

@app.before_request
def before_request():
    g.start_time = time.time()
    service_health['last_request'] = datetime.utcnow()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        response.headers['X-Response-Time'] = f'{elapsed:.3f}s'

        if elapsed > 5.0:
            logger.warning(f"Slow request: {request.path} took {elapsed:.2f}s")
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    return response

@app.route('/health', methods=['GET'])
@app.route('/_ah/health', methods=['GET'])
def health_check():
    try:
        uptime = (datetime.utcnow() - service_health['start_time']).total_seconds()
        total_requests = service_health['requests_processed']
        failed_requests = service_health['requests_failed']

        success_rate = 100
        if total_requests > 0:
            success_rate = ((total_requests - failed_requests) / total_requests) * 100

        health_data = {
            'status': 'healthy',
            'service': 'satyacheck-ai',
            'version': '2.0-hackathon',
            'uptime_seconds': int(uptime),
            'requests_processed': total_requests,
            'success_rate': f'{success_rate:.1f}%',
            'timestamp': datetime.utcnow().isoformat()
        }

        return health_data, 200

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {'status': 'unhealthy', 'error': str(e)}, 503

@app.route('/readiness', methods=['GET'])
def readiness_check():
    try:
        return {'status': 'ready', 'timestamp': datetime.utcnow().isoformat()}, 200
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {'status': 'not_ready', 'error': str(e)}, 503

@app.route('/ivr/dashboard', methods=['GET'])
@app.route('/ivr/analytics/dashboard', methods=['GET'])
def ivr_dashboard():
    try:
        return send_file('ivr_analytics_dashboard.html')
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return {'error': 'Dashboard not available'}, 500

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    try:
        service_health['requests_processed'] += 1
        result = satyacheck_main(request)

        if isinstance(result, tuple):
            if len(result) == 2:
                response, status_code = result
                if status_code >= 500:
                    service_health['requests_failed'] += 1
                return response, status_code
            elif len(result) == 3:
                body, status_code, headers = result
                if status_code >= 500:
                    service_health['requests_failed'] += 1
                return body, status_code, headers

        return result

    except Exception as e:
        service_health['requests_failed'] += 1
        logger.error(f"Error in catch_all route: {e}", exc_info=True)
        return {
            'success': False,
            'error': 'internal_error',
            'message': 'An unexpected error occurred'
        }, 500

@app.errorhandler(404)
def not_found(error):
    return {'success': False, 'error': 'not_found', 'message': 'Endpoint not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}", exc_info=True)
    return {'success': False, 'error': 'internal_error', 'message': 'Internal server error'}, 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return {'success': False, 'error': 'file_too_large', 'message': 'File too large. Maximum size is 50MB'}, 413

try:
    from whatsapp_bot import whatsapp_bp
    app.register_blueprint(whatsapp_bp)
    logger.info("WhatsApp bot registered")
except Exception as e:
    logger.warning(f"WhatsApp bot not registered: {e}")

try:
    from ivr_bot import ivr_bp
    app.register_blueprint(ivr_bp)
    logger.info("IVR system registered")
except Exception as e:
    logger.warning(f"IVR system not registered: {e}")

try:
    from ivr_advanced import ivr_advanced_bp
    app.register_blueprint(ivr_advanced_bp)
    logger.info("Advanced IVR registered")
except Exception as e:
    logger.warning(f"Advanced IVR not registered: {e}")

try:
    from deepfake_api import deepfake_bp
    app.register_blueprint(deepfake_bp)
    logger.info("Deepfake API registered")
except Exception as e:
    logger.warning(f"Deepfake API not registered: {e}")

def startup():
    logger.info("=" * 80)
    logger.info("SATYACHECK AI - STARTING UP")
    logger.info("=" * 80)
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    logger.info(f"Project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'unknown')}")
    logger.info(f"Request Timeout: {REQUEST_TIMEOUT}s")
    logger.info("=" * 80)

startup()

def handle_shutdown(signum, frame):
    logger.info("=" * 80)
    logger.info("GRACEFUL SHUTDOWN INITIATED")
    logger.info("=" * 80)
    uptime = (datetime.utcnow() - service_health['start_time']).total_seconds()
    logger.info(f"Total uptime: {uptime:.0f}s")
    logger.info(f"Requests processed: {service_health['requests_processed']}")
    logger.info(f"Requests failed: {service_health['requests_failed']}")
    logger.info("=" * 80)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
