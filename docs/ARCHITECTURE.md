# Architecture Documentation - SatyaCheck AI

Technical architecture and design decisions for the SatyaCheck AI platform.

## System Overview

SatyaCheck AI is a multi-channel fact-checking and deepfake detection platform built on Google Cloud Platform, powered by Google Gemini AI.

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        User Channels                          │
├───────────────┬────────────────┬──────────────┬───────────────┤
│   Web UI      │   WhatsApp     │   IVR/Voice  │   API Direct  │
│   (Browser)   │   (Twilio)     │   (Twilio)   │   (REST)      │
└───────┬───────┴────────┬───────┴──────┬───────┴───────┬───────┘
        │                │              │               │
        └────────────────┴──────────────┴───────────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │      Load Balancer / API Gateway         │
        │         (Google Cloud Run)               │
        └─────────────────────┬─────────────────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │           Main Backend Service            │
        │          (Flask on Cloud Run)             │
        │                                           │
        │  ┌─────────────────────────────────────┐ │
        │  │     Request Router & Validator      │ │
        │  └──────────────┬──────────────────────┘ │
        │                 │                         │
        │  ┌──────────────▼──────────────────────┐ │
        │  │      Content Processor              │ │
        │  │  - Text extraction                  │ │
        │  │  - Image processing                 │ │
        │  │  - Audio transcription              │ │
        │  │  - Video frame extraction           │ │
        │  └──────────────┬──────────────────────┘ │
        └─────────────────┼─────────────────────────┘
                          │
        ┌─────────────────▼─────────────────────┐
        │         Analysis Layer                │
        ├───────────────────┬───────────────────┤
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Fact-Check  │   │  Deepfake    │   │ Multi-Claim  │
│   Engine     │   │  Detector    │   │  Analyzer    │
│              │   │              │   │              │
│ • Gemini AI  │   │ • Image ML   │   │ • Claim      │
│ • PIB DB     │   │ • Audio ML   │   │   Extraction │
│ • Web Search │   │ • Video ML   │   │ • Parallel   │
│ • Context    │   │ • Forensics  │   │   Analysis   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
        ┌─────────────────▼─────────────────────┐
        │        External Services              │
        ├───────────────┬───────────────────────┤
        │ Gemini 2.0    │ Google Search API     │
        │ PIB Database  │ Reverse Image Search  │
        │ Twilio Voice  │ Twilio WhatsApp       │
        └───────────────┴───────────────────────┘
                          │
        ┌─────────────────▼─────────────────────┐
        │          Data Storage                 │
        ├───────────────┬───────────────────────┤
        │ Firestore     │ Cloud Storage         │
        │ (metadata,    │ (temporary files,     │
        │  cache,       │  uploaded media)      │
        │  analytics)   │                       │
        └───────────────┴───────────────────────┘
```

## Component Details

### 1. Frontend Layer

**Technologies:**
- HTML5, CSS3, JavaScript (vanilla)
- Responsive design (mobile-first)
- Progressive enhancement

**Responsibilities:**
- User input collection
- File upload handling
- API communication
- Results presentation
- Error handling

**Deployment:**
- Static hosting (Firebase/GCS)
- CDN for global distribution
- HTTPS enforced

### 2. API Gateway (Cloud Run)

**Technologies:**
- Google Cloud Run
- Container-based deployment
- Auto-scaling

**Features:**
- Request routing
- Load balancing
- SSL termination
- DDoS protection
- Rate limiting

**Configuration:**
```yaml
Memory: 2Gi
CPU: 2
Concurrency: 80
Min instances: 1
Max instances: 100
Timeout: 300s
```

### 3. Backend Service

**Technologies:**
- Python 3.11
- Flask web framework
- Gunicorn WSGI server

**Architecture Pattern:**
- Layered architecture
- Separation of concerns
- Dependency injection
- Error boundary pattern

**Modules:**

**main.py** - Entry point
```python
@app.route('/', methods=['POST'])
def analyze_content():
    # 1. Validate input
    # 2. Route to appropriate handler
    # 3. Format response
    # 4. Return result
```

**deepfake_detector.py** - ML inference
```python
class DeepfakeDetector:
    # Lazy loading of models
    # GPU/CPU detection
    # Batch processing
    # Result caching
```

**ai_analyzer.py** - Gemini integration
```python
async def analyze_claim_with_ai(claim, language):
    # Structured prompts
    # Response parsing
    # Error recovery
    # Retry logic
```

### 4. Analysis Layer

#### A. Fact-Checking Engine

**Components:**
1. **Claim Extraction**
   - NLP-based extraction
   - Multi-claim detection
   - Entity recognition

2. **Evidence Collection**
   - Web search (Google API)
   - PIB fact-check database
   - Academic sources
   - News verification

3. **Credibility Scoring**
   ```python
   credibility_score = (
       source_reputation * 0.4 +
       evidence_quality * 0.3 +
       temporal_relevance * 0.2 +
       cross_references * 0.1
   )
   ```

4. **Verdict Generation**
   - TRUE / FALSE / PARTIALLY TRUE
   - Confidence percentage
   - Evidence summary
   - Source citations

#### B. Deepfake Detection

**Image Detection:**
```
Input Image
    ↓
Preprocessing (resize, normalize)
    ↓
Vision Transformer (ViT)
    ↓
CLIP Model
    ↓
Ensemble Decision
    ↓
Confidence Score
```

**Models:**
- ViT: `google/vit-base-patch16-224`
- CLIP: `openai/clip-vit-base-patch32`

**Audio Detection:**
```
Input Audio
    ↓
Feature Extraction (MFCC, Spectral)
    ↓
Wav2Vec2 Model
    ↓
Forensic Analysis
    ↓
Ensemble Score
```

**Model:**
- Wav2Vec2: `MelodyMachine/Deepfake-audio-detection-V2`
- Accuracy: 99.73%

**Video Detection:**
```
Input Video
    ↓
Frame Extraction (30 frames)
    ↓
Per-frame Analysis
    ↓
Temporal Consistency Check
    ↓
Face Tracking
    ↓
Aggregated Verdict
```

**Algorithms:**
- Frame sampling: Uniform distribution
- Temporal analysis: Inter-frame correlation
- Face detection: OpenCV Haar Cascades

### 5. External Services Integration

#### Google Gemini AI

**Configuration:**
```python
model = genai.GenerativeModel('gemini-2.0-flash-exp')
generation_config = {
    'temperature': 0.3,
    'top_p': 0.95,
    'top_k': 40,
    'max_output_tokens': 8192
}
```

**Prompt Engineering:**
```
System: You are a fact-checking expert...
Context: [relevant context]
Claim: [user claim]
Task: Analyze and provide structured response
```

#### Twilio Integration

**Voice (IVR):**
```python
TwiML → Speech-to-Text → Analysis → Text-to-Speech → SMS
```

**WhatsApp:**
```python
Webhook → Message Processing → Analysis → Response
```

### 6. Data Storage

#### Firestore (NoSQL)

**Collections:**
```
/analyses
  /{analysis_id}
    - claim_hash
    - timestamp
    - result
    - cache_ttl

/users (optional)
  /{user_id}
    - query_history
    - preferences

/analytics
  /{date}
    - metrics
```

#### Cloud Storage

**Buckets:**
```
{project-id}-temp-files/
  - Lifecycle: Delete after 1 day
  - Purpose: Temporary media storage

{project-id}-models/
  - Purpose: ML model checkpoints
```

## Data Flow

### Text Fact-Check Flow

```
1. User submits text → Frontend
2. Frontend → POST /api → Backend
3. Backend validates input
4. Extract key terms and claims
5. Search web (Google API)
6. Query PIB database
7. Analyze with Gemini AI
8. Calculate credibility score
9. Format response
10. Cache result (Firestore)
11. Return to user
```

**Latency Budget:**
- Input validation: <100ms
- Search queries: <1s
- AI analysis: 2-4s
- Total: <5s

### Image Deepfake Detection Flow

```
1. User uploads image → Frontend
2. Frontend → POST /api (multipart) → Backend
3. Backend validates file (type, size)
4. Save to temp storage
5. Preprocess image (resize, normalize)
6. Load ML models (lazy)
7. Run inference (ViT + CLIP)
8. Ensemble scoring
9. Format result with indicators
10. Cleanup temp file
11. Return verdict
```

**Latency Budget:**
- Upload: <1s
- Processing: 1-2s
- Total: <3s

## Scaling Strategy

### Horizontal Scaling

**Cloud Run Auto-scaling:**
```
Triggers:
- CPU utilization > 70%
- Request queue depth > 20
- Custom metrics

Scale up: +10 instances every 30s
Scale down: -1 instance every 5 minutes
```

### Caching Strategy

**Multi-Level Caching:**

1. **In-Memory Cache (Application)**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def get_analysis(claim_hash):
       # Fast lookup for repeated claims
   ```

2. **Distributed Cache (Redis - Future)**
   ```python
   # For multi-instance coordination
   redis.get(f'analysis:{claim_hash}')
   ```

3. **Database Cache (Firestore)**
   ```python
   # Persistent cache with TTL
   db.collection('analyses').document(claim_hash).get()
   ```

**Cache Strategy:**
- Cache Key: SHA-256 hash of normalized claim
- TTL: 1 hour (configurable)
- Invalidation: Manual + time-based

### Database Optimization

**Firestore Indexing:**
```
Indexes:
- claim_hash (for lookups)
- timestamp (for analytics)
- user_id + timestamp (for history)
```

**Query Optimization:**
```python
# Use cached reads
db.collection('analyses').where('claim_hash', '==', hash).limit(1).get()
```

## Security Architecture

### Defense in Depth

**Layer 1: Network**
- HTTPS only (TLS 1.3)
- Cloud Armor (DDoS protection)
- VPC for internal communication

**Layer 2: Application**
- Input validation
- Content sanitization
- SQL injection prevention
- XSS prevention

**Layer 3: Authentication**
- API key validation (future)
- Rate limiting per IP
- Bot detection

**Layer 4: Data**
- Encryption at rest (GCP default)
- Encryption in transit (TLS)
- Secret Manager for credentials

### Content Safety

**Filters:**
```python
def check_content_safety(text):
    # Check for harmful content
    # Profanity filter
    # NSFW detection (images)
    # Violence detection (videos)
```

**Moderation:**
- Automated flagging
- Review queue (future)
- User reporting

## Monitoring & Observability

### Logging

**Structured Logging:**
```python
import logging
logger.info('Processing request', extra={
    'request_id': req_id,
    'user_ip': ip,
    'input_type': type,
    'processing_time': duration
})
```

**Log Levels:**
- DEBUG: Development only
- INFO: Normal operations
- WARNING: Degraded performance
- ERROR: Request failures

### Metrics

**Key Metrics:**
```
- Requests per second
- Response time (p50, p95, p99)
- Error rate
- Cache hit rate
- Model inference time
- Cost per request
```

**Dashboards:**
- Real-time metrics (Cloud Monitoring)
- Custom dashboards (Grafana)
- Alerting (PagerDuty/Email)

### Tracing

**Distributed Tracing:**
```python
from opencensus.trace import tracer
from opencensus.ext.flask.flask_middleware import FlaskMiddleware

# Trace requests across services
with tracer.span(name='fact_check'):
    result = analyze_claim(claim)
```

## Performance Optimization

### Model Optimization

**Techniques:**
1. **Lazy Loading**
   ```python
   # Load models only when needed
   if not self.model:
       self.model = load_model()
   ```

2. **Model Quantization**
   ```python
   # Reduce model size and inference time
   torch.quantization.quantize_dynamic(model)
   ```

3. **Batch Processing**
   ```python
   # Process multiple inputs together
   results = model(batch_inputs)
   ```

### API Optimization

**Techniques:**
1. **Request Batching**
   - Combine multiple API calls
   - Reduce network overhead

2. **Connection Pooling**
   ```python
   session = requests.Session()
   adapter = HTTPAdapter(pool_connections=100)
   ```

3. **Async Processing**
   ```python
   async def analyze_multiple_claims(claims):
       tasks = [analyze_claim(c) for c in claims]
       return await asyncio.gather(*tasks)
   ```

## Disaster Recovery

### Backup Strategy

**Automated Backups:**
- Firestore: Daily export to GCS
- Container images: Versioned in GCR
- Secrets: Versioned in Secret Manager

**Recovery Procedures:**
```bash
# Restore database
gcloud firestore import gs://backups/latest

# Rollback deployment
gcloud run services update-traffic --to-revisions=prev
```

### High Availability

**Multi-Region (Future):**
```
Primary: us-central1
Secondary: europe-west1
Failover: Automatic (Cloud Load Balancing)
```

## Cost Optimization

### Current Costs (Estimated)

**Per 1000 requests:**
- Cloud Run: $0.10
- Gemini API: $0.50
- Storage: $0.01
- Network: $0.05
- **Total: ~$0.66**

### Optimization Strategies

1. **Caching** - Reduce AI API calls by 40%
2. **Right-sizing** - Match resources to usage
3. **Spot instances** - Use preemptible VMs (future)
4. **Compression** - Reduce network costs

## Future Enhancements

### Planned Features

1. **Real-time Updates**
   - WebSocket support
   - Live fact-checking
   - Streaming responses

2. **Advanced Analytics**
   - User behavior tracking
   - Claim trending
   - Misinformation patterns

3. **Blockchain Integration**
   - Immutable audit trail
   - Verifiable credentials
   - Decentralized storage

4. **Multi-Modal Models**
   - Combined image+text analysis
   - Video understanding
   - Cross-modal verification

### Scalability Roadmap

**Phase 1 (Current):** 100 RPS
**Phase 2 (Q2):** 1,000 RPS
**Phase 3 (Q3):** 10,000 RPS
**Phase 4 (Q4):** 100,000 RPS

## Technology Decisions

### Why Google Gemini?

- State-of-the-art AI capabilities
- Multi-modal understanding
- Cost-effective pricing
- Hackathon requirement

### Why Cloud Run?

- Serverless (no infrastructure management)
- Auto-scaling
- Pay-per-use
- Fast cold starts

### Why Python?

- Rich ML ecosystem
- Easy integration
- Rapid development
- Community support

### Why Firestore?

- NoSQL flexibility
- Real-time capabilities
- Automatic scaling
- GCP integration

## References

- [Cloud Run Best Practices](https://cloud.google.com/run/docs/best-practices)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [ML Model Optimization](https://www.tensorflow.org/model_optimization)
