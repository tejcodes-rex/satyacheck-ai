# SatyaCheck AI - Multi-Channel Fact-Checking Platform

**Powered by Google Gemini AI**

SatyaCheck AI is a comprehensive fact-checking platform that combats misinformation through multiple channels including web interface, WhatsApp, and IVR (Interactive Voice Response). The system leverages advanced AI and deepfake detection to verify claims and multimedia content in real-time.

## Hackathon Submission

This project is submitted for the **Google Gen AI Exchange Hackathon** by **Team Veyra** (Team Leader: Tejas Mane).

## Live Demo

- **Web Application:** https://satyacheck-ai-290073510140.us-central1.run.app/
- **Backend API:** https://satyacheck-ai-290073510140.us-central1.run.app/
- **Deepfake Detection:** 100% accuracy on test files (verified)

## Key Features

### 1. Multi-Channel Access
- **Web Interface** - Simple, responsive web UI for desktop and mobile
- **WhatsApp Bot** - Fact-checking via WhatsApp messaging
- **IVR System** - Voice-based fact-checking in 7+ Indian languages

### 2. Advanced AI Capabilities
- **Google Gemini Integration** - Powered by Gemini 2.0 Flash Exp for intelligent analysis
- **Deepfake Detection** - Audio, Image, and Video deepfake detection with ML models
- **Multi-Language Support** - Hindi, English, Tamil, Telugu, Bengali, Marathi, Gujarati, and more
- **Real-time Fact-Checking** - Instant verification with credibility scoring

### 3. Deepfake Detection Performance
**100% Accuracy** on test dataset:
- Image Detection: 2/2 tests passed (avg 1.30s)
- Audio Detection: 2/2 tests passed (avg 1.42s)
- Video Detection: 2/2 tests passed (avg 3.13s)

### 4. Intelligent Features
- Multi-claim detection and analysis
- Reverse image search for context
- Source credibility scoring
- Temporal and contextual analysis
- Cultural context awareness
- Content safety filters

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │   Web    │    │ WhatsApp │    │   IVR    │             │
│  │Interface │    │   Bot    │    │  (Voice) │             │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘             │
└───────┼───────────────┼───────────────┼────────────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      Backend API (Flask)       │
        │  - Request routing             │
        │  - Input validation            │
        │  - Response formatting         │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     AI Analysis Layer          │
        │  ┌─────────────────────────┐  │
        │  │   Google Gemini 2.0     │  │
        │  │   - Fact verification   │  │
        │  │   - Source analysis     │  │
        │  │   - Claim extraction    │  │
        │  └─────────────────────────┘  │
        │  ┌─────────────────────────┐  │
        │  │  Deepfake Detection     │  │
        │  │   - Audio (Wav2Vec2)    │  │
        │  │   - Image (ViT+CLIP)    │  │
        │  │   - Video (Frame-based) │  │
        │  └─────────────────────────┘  │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │    External Services           │
        │  - Google Search API           │
        │  - PIB Fact Check DB           │
        │  - Reverse Image Search        │
        │  - Twilio (Voice/SMS)          │
        └────────────────────────────────┘
```

## Project Structure

```
satyacheck-hackathon-submission/
├── backend/                    # Core backend services
│   ├── main.py                 # Main API handler
│   ├── deepfake_detector.py   # Deepfake detection engine
│   ├── ai_analyzer.py          # Gemini AI integration
│   ├── config.py               # Configuration settings
│   ├── utils.py                # Utility functions
│   ├── database.py             # Database operations
│   └── ...                     # Other backend modules
│
├── ivr-bot/                    # IVR System
│   ├── ivr_bot.py              # Basic IVR implementation
│   ├── ivr_advanced.py         # Advanced IVR features
│   ├── ivr_sms_notifications.py # SMS integration
│   └── ivr_analytics_dashboard.html # Analytics UI
│
├── whatsapp-bot/               # WhatsApp Integration
│   └── whatsapp_bot.py         # WhatsApp bot handler
│
├── frontend/                   # Web Interface
│   └── index.html              # Main web UI
│
├── deployment-configs/         # Deployment files
│   ├── Dockerfile              # Container configuration
│   ├── requirements.txt        # Python dependencies
│   └── cloudbuild.yaml         # Google Cloud Build config
│
├── test-files/                 # Test suite
│   ├── test_all_deepfakes.py  # Comprehensive test script
│   ├── deployed_test_results.json # Latest test results
│   └── deepfake-*.*            # Test media files
│
└── docs/                       # Documentation
    └── (component-specific docs)
```

## Technology Stack

### AI & ML
- **Google Gemini 2.0 Flash Exp** - Primary AI model for fact-checking
- **Wav2Vec2** - Audio deepfake detection (99.73% accuracy)
- **Vision Transformer (ViT)** - Image analysis
- **CLIP** - Image understanding and context

### Backend
- **Python 3.11+** - Core programming language
- **Flask** - Web framework
- **Google Cloud Run** - Serverless deployment
- **Google Cloud Storage** - File storage
- **Firestore** - NoSQL database

### Communication Channels
- **Twilio Voice API** - IVR system
- **Twilio WhatsApp API** - WhatsApp integration
- **HTTP/REST** - Web API

### External APIs
- **Google Search API** - Web search for fact verification
- **PIB Fact Check** - Government fact-check database
- **Reverse Image Search** - Image context verification

## Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud account (with Gemini API access)
- Twilio account (for IVR/WhatsApp features)

### Environment Variables
```bash
# Google Cloud
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_SEARCH_API_KEY=your_search_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
GCP_PROJECT_ID=your_project_id

# Twilio (Optional - for IVR/WhatsApp)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
TWILIO_WHATSAPP_NUMBER=your_whatsapp_number
```

### Local Development

1. **Install dependencies:**
```bash
cd backend
pip install -r ../deployment-configs/requirements.txt
```

2. **Set environment variables:**
```bash
# Windows
set GOOGLE_API_KEY=your_api_key

# Linux/Mac
export GOOGLE_API_KEY=your_api_key
```

3. **Run the server:**
```bash
python main.py
```

4. **Test the API:**
```bash
curl -X POST http://localhost:8080/ \
  -F "text_data=Is the earth flat?" \
  -F "language=en"
```

### Deployment to Google Cloud Run

```bash
cd deployment-configs
gcloud builds submit --config cloudbuild.yaml
```

## Testing

### Run Deepfake Detection Tests
```bash
cd test-files
python test_all_deepfakes.py
```

**Latest Test Results (100% Accuracy):**
- Total Tests: 6/6 passed
- Image Detection: 2/2 passed
- Audio Detection: 2/2 passed
- Video Detection: 2/2 passed

## Usage Examples

### 1. Web Interface
1. Visit the web application URL
2. Enter text, upload image/audio/video, or paste URL
3. Select language (optional)
4. Enable deep analysis for deepfake detection
5. Click "Check Claim"
6. View results with credibility score and sources

### 2. WhatsApp Bot
1. Send a message to the WhatsApp number
2. Send text claim, image, audio, or video
3. Receive instant fact-check results
4. View credibility scores and evidence

### 3. IVR System
1. Call the IVR number
2. Select your preferred language (1-7)
3. Speak your claim after the beep
4. Listen to the AI-powered fact-check results
5. Receive SMS with detailed results and sources

## Key Capabilities

### Fact-Checking
- **Text Claims** - Verify statements and news claims
- **URLs** - Extract and analyze content from web pages
- **Multi-Claim** - Detect and verify multiple claims in one input

### Deepfake Detection
- **Image Analysis** - Detect manipulated images with ML models
- **Audio Analysis** - Identify synthetic/cloned voices
- **Video Analysis** - Frame-by-frame deepfake detection

### Language Support
- English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati
- Automatic language detection
- Multi-language response generation

### Source Verification
- Credibility scoring based on source reputation
- Cross-referencing with trusted fact-checking databases
- Evidence collection and citation

## Performance Metrics

### Deepfake Detection (Deployed System)
- **Image Processing:** ~1.30s average
- **Audio Processing:** ~1.42s average
- **Video Processing:** ~3.13s average (client-side)
- **Accuracy:** 100% on test dataset

### Fact-Checking
- **Average Response Time:** 2-5 seconds
- **Multi-language Support:** 7+ languages
- **Source Verification:** Multiple trusted sources
- **Caching:** Instant responses for repeated queries

## Security & Privacy

- Content safety filters for harmful content
- Secure file handling with automatic cleanup
- Privacy-focused design (no permanent storage without consent)
- Rate limiting and abuse prevention
- Sanitized error messages

## Team Information

**Team Name:** Team Veyra
**Team Leader:** Tejas Mane

Developed for the **Google Gen AI Exchange Hackathon**

## License

This project is developed for the hackathon submission. All rights reserved.

## Acknowledgments

- **Google Gemini AI** - Core AI capabilities
- **Twilio** - Voice and messaging infrastructure
- **HuggingFace** - ML models for deepfake detection
- **Google Cloud Platform** - Hosting and infrastructure

## Support & Contact

For questions or issues related to this hackathon submission, please refer to the documentation in the `docs/` directory.

---

**Built with Google Gemini AI | Combating Misinformation | Empowering Truth**
