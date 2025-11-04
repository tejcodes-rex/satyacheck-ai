# Deployment Guide - SatyaCheck AI

Complete deployment guide for Google Cloud Platform.

## Prerequisites

### Required Accounts
- Google Cloud Platform account
- Google Gemini API access
- Twilio account (for IVR/WhatsApp)

### Required Tools
```bash
# Google Cloud SDK
gcloud --version

# Docker
docker --version

# Python 3.11+
python --version
```

## Environment Setup

### 1. Google Cloud Project

```bash
# Set project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Secret Manager

Store sensitive credentials securely:

```bash
# Google API Key (Gemini)
echo -n "your-gemini-api-key" | \
  gcloud secrets create google-api-key --data-file=-

# Google Search API Key
echo -n "your-search-api-key" | \
  gcloud secrets create google-search-api-key --data-file=-

# Google Search Engine ID
echo -n "your-engine-id" | \
  gcloud secrets create google-search-engine-id --data-file=-

# Twilio credentials (optional)
echo -n "your-twilio-sid" | \
  gcloud secrets create twilio-account-sid --data-file=-

echo -n "your-twilio-token" | \
  gcloud secrets create twilio-auth-token --data-file=-
```

### 3. Firestore Database

```bash
# Create Firestore database
gcloud firestore databases create --region=us-central1

# Import security rules (optional)
gcloud firestore deploy --rules=firestore.rules
```

### 4. Cloud Storage

```bash
# Create bucket for temporary files
gsutil mb -p $PROJECT_ID -l us-central1 gs://${PROJECT_ID}-temp-files

# Set lifecycle policy (auto-delete after 1 day)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 1}
    }]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-temp-files
```

## Backend Deployment

### Option 1: Cloud Build (Recommended)

```bash
cd deployment-configs

# Deploy using Cloud Build
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_PROJECT_ID=$PROJECT_ID

# Wait for deployment (5-10 minutes)
```

### Option 2: Manual Deployment

```bash
# Build container
docker build -t gcr.io/$PROJECT_ID/satyacheck-ai:latest -f Dockerfile .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/satyacheck-ai:latest

# Deploy to Cloud Run
gcloud run deploy satyacheck-ai \
  --image gcr.io/$PROJECT_ID/satyacheck-ai:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1 \
  --allow-unauthenticated \
  --set-secrets="GOOGLE_API_KEY=google-api-key:latest,GOOGLE_SEARCH_API_KEY=google-search-api-key:latest,GOOGLE_SEARCH_ENGINE_ID=google-search-engine-id:latest" \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID"
```

### Get Deployment URL

```bash
# Get service URL
gcloud run services describe satyacheck-ai \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'

# Test deployment
curl https://your-service-url/health
```

## Frontend Deployment

### Option 1: Firebase Hosting

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize Firebase
cd frontend
firebase init hosting

# Configure firebase.json
cat > firebase.json << EOF
{
  "hosting": {
    "public": ".",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [{
      "source": "**",
      "destination": "/index.html"
    }]
  }
}
EOF

# Update API URL in index.html
sed -i 's|API_URL = .*|API_URL = "https://your-backend-url"|' index.html

# Deploy
firebase deploy --only hosting
```

### Option 2: Cloud Storage Static Website

```bash
# Create bucket
gsutil mb gs://${PROJECT_ID}-frontend

# Make bucket public
gsutil iam ch allUsers:objectViewer gs://${PROJECT_ID}-frontend

# Set as website
gsutil web set -m index.html -e index.html gs://${PROJECT_ID}-frontend

# Upload files
gsutil -m cp -r frontend/* gs://${PROJECT_ID}-frontend/

# Access at: http://storage.googleapis.com/${PROJECT_ID}-frontend/index.html
```

## IVR Deployment

### 1. Deploy IVR Handler

```bash
# Deploy IVR as separate Cloud Run service
gcloud run deploy satyacheck-ivr \
  --image gcr.io/$PROJECT_ID/satyacheck-ai:latest \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --allow-unauthenticated \
  --set-secrets="TWILIO_ACCOUNT_SID=twilio-account-sid:latest,TWILIO_AUTH_TOKEN=twilio-auth-token:latest"
```

### 2. Configure Twilio Webhook

```bash
# Get IVR service URL
IVR_URL=$(gcloud run services describe satyacheck-ivr \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

# Configure in Twilio Console:
# Voice & Fax > A Call Comes In
# Webhook: ${IVR_URL}/ivr/incoming
# HTTP Method: POST
```

## WhatsApp Deployment

### 1. Configure Twilio WhatsApp

```bash
# Get backend URL
BACKEND_URL=$(gcloud run services describe satyacheck-ai \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

# Configure in Twilio Console:
# Messaging > WhatsApp Sandbox Settings
# When a message comes in: ${BACKEND_URL}/whatsapp/incoming
# HTTP Method: POST
```

### 2. Test WhatsApp Integration

```bash
# Send test message via Twilio
curl -X POST https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Messages.json \
  --data-urlencode "From=whatsapp:${TWILIO_NUMBER}" \
  --data-urlencode "To=whatsapp:+1234567890" \
  --data-urlencode "Body=Test message" \
  -u ${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}
```

## Configuration

### Environment Variables

Update in Cloud Run:
```bash
gcloud run services update satyacheck-ai \
  --region us-central1 \
  --set-env-vars="KEY=VALUE"
```

### Secrets

Update secrets:
```bash
# Update existing secret
echo -n "new-value" | gcloud secrets versions add secret-name --data-file=-

# Cloud Run automatically uses latest version
```

## Monitoring & Logging

### View Logs

```bash
# Backend logs
gcloud run services logs read satyacheck-ai \
  --region us-central1 \
  --limit 100

# Follow logs in real-time
gcloud run services logs tail satyacheck-ai \
  --region us-central1
```

### Set Up Alerts

```bash
# Create alert for errors
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-threshold-value=10 \
  --condition-threshold-duration=60s
```

### Performance Monitoring

```bash
# Cloud Monitoring Dashboard
gcloud monitoring dashboards create --config-from-file=dashboard.yaml
```

## Scaling Configuration

### Auto-scaling

```bash
gcloud run services update satyacheck-ai \
  --region us-central1 \
  --min-instances 1 \
  --max-instances 100 \
  --concurrency 80
```

### Resource Limits

```bash
# Update memory and CPU
gcloud run services update satyacheck-ai \
  --region us-central1 \
  --memory 4Gi \
  --cpu 4
```

## Cost Optimization

### 1. Minimize Cold Starts
```bash
# Set minimum instances
--min-instances 1  # Keeps 1 instance warm
```

### 2. Optimize Container
```dockerfile
# Use smaller base image
FROM python:3.11-slim

# Multi-stage build
FROM python:3.11 as builder
# ... build steps
FROM python:3.11-slim
COPY --from=builder /app /app
```

### 3. Enable Request Caching
- Implement in-memory caching
- Use Redis for distributed cache
- Set appropriate TTL values

### 4. Resource Right-sizing
```bash
# Monitor actual usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/memory/utilizations"'

# Adjust based on metrics
```

## Security Best Practices

### 1. IAM Roles

```bash
# Create service account
gcloud iam service-accounts create satyacheck-sa

# Grant minimal permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:satyacheck-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 2. VPC Configuration

```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create satyacheck-connector \
  --region us-central1 \
  --range 10.8.0.0/28

# Use with Cloud Run
gcloud run services update satyacheck-ai \
  --vpc-connector satyacheck-connector
```

### 3. Enable Binary Authorization

```bash
gcloud services enable binaryauthorization.googleapis.com

# Create policy
gcloud container binauthz policy import policy.yaml
```

## Backup & Recovery

### Database Backup

```bash
# Export Firestore
gcloud firestore export gs://${PROJECT_ID}-backups

# Schedule automated backups
gcloud scheduler jobs create http firestore-backup \
  --schedule="0 2 * * *" \
  --uri="https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default):exportDocuments" \
  --message-body='{"outputUriPrefix":"gs://'${PROJECT_ID}'-backups"}'
```

### Disaster Recovery

```bash
# Import from backup
gcloud firestore import gs://${PROJECT_ID}-backups/[EXPORT_PREFIX]

# Rollback to previous version
gcloud run services update satyacheck-ai \
  --region us-central1 \
  --image gcr.io/$PROJECT_ID/satyacheck-ai:v1.0.0
```

## Testing Deployment

### Automated Tests

```bash
cd test-files

# Update backend URL
export BACKEND_URL="https://your-deployed-url"

# Run tests
python test_all_deepfakes.py
```

### Manual Tests

```bash
# Health check
curl https://your-backend-url/health

# Text claim
curl -X POST https://your-backend-url/ \
  -F "text_data=Test claim" \
  -F "language=en"

# Image upload
curl -X POST https://your-backend-url/ \
  -F "image_data=@test-image.jpg" \
  -F "deep_analysis=true"
```

## Troubleshooting

### Common Issues

**1. Container fails to start**
```bash
# Check build logs
gcloud builds list --limit=5

# Check runtime logs
gcloud run services logs read satyacheck-ai --limit=50
```

**2. Secrets not accessible**
```bash
# Verify secret exists
gcloud secrets list

# Check IAM permissions
gcloud secrets get-iam-policy google-api-key
```

**3. Out of memory errors**
```bash
# Increase memory
gcloud run services update satyacheck-ai --memory 4Gi
```

**4. Slow response times**
```bash
# Check cold start times
gcloud run services describe satyacheck-ai

# Increase min instances
gcloud run services update satyacheck-ai --min-instances 2
```

## Rollback

```bash
# List revisions
gcloud run revisions list --service satyacheck-ai

# Rollback to previous revision
gcloud run services update-traffic satyacheck-ai \
  --to-revisions REVISION-NAME=100
```

## CI/CD Pipeline

### GitHub Actions

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: google-github-actions/setup-gcloud@v0
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
      - run: gcloud builds submit --config cloudbuild.yaml
```

## Maintenance

### Regular Tasks
- [ ] Review logs weekly
- [ ] Update dependencies monthly
- [ ] Check security alerts
- [ ] Monitor costs
- [ ] Test backups quarterly
- [ ] Review scaling metrics
- [ ] Update documentation

## Support Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Troubleshooting Guide](https://cloud.google.com/run/docs/troubleshooting)
- [Best Practices](https://cloud.google.com/run/docs/best-practices)
