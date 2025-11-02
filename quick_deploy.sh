#!/bin/bash
# SatyaCheck AI - Quick Deploy Script
# This script automates the entire deployment process

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════╗"
echo "║  SatyaCheck AI - Cloud Deployment     ║"
echo "║  Automated Setup & Deploy              ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: Google Cloud CLI (gcloud) is not installed${NC}"
    echo -e "${YELLOW}Install from: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Get project configuration
echo -e "\n${YELLOW}═══ Project Configuration ═══${NC}"
read -p "Enter your Google Cloud Project ID (or press Enter for 'satyacheck-ai'): " PROJECT_ID
PROJECT_ID=${PROJECT_ID:-"satyacheck-ai"}

read -p "Enter region (or press Enter for 'us-central1'): " REGION
REGION=${REGION:-"us-central1"}

read -p "Enter service name (or press Enter for 'satyacheck-ai'): " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-"satyacheck-ai"}

echo -e "\n${BLUE}Configuration:${NC}"
echo "  Project ID:   $PROJECT_ID"
echo "  Region:       $REGION"
echo "  Service Name: $SERVICE_NAME"

read -p "$(echo -e ${YELLOW}Is this correct? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Deployment cancelled${NC}"
    exit 1
fi

# Set project
echo -e "\n${GREEN}Step 1/7: Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID 2>/dev/null || {
    echo -e "${YELLOW}Project doesn't exist. Creating new project...${NC}"
    gcloud projects create $PROJECT_ID --name="SatyaCheck AI" || {
        echo -e "${RED}Failed to create project. You may need to create it manually at:${NC}"
        echo -e "${YELLOW}https://console.cloud.google.com/projectcreate${NC}"
        exit 1
    }
    gcloud config set project $PROJECT_ID
}

# Enable APIs
echo -e "\n${GREEN}Step 2/7: Enabling required APIs...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

gcloud services enable run.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet
gcloud services enable vision.googleapis.com --quiet
gcloud services enable speech.googleapis.com --quiet
gcloud services enable translate.googleapis.com --quiet
gcloud services enable texttospeech.googleapis.com --quiet
gcloud services enable secretmanager.googleapis.com --quiet
gcloud services enable firestore.googleapis.com --quiet

echo -e "${GREEN}✓ APIs enabled${NC}"

# Check for API keys
echo -e "\n${GREEN}Step 3/7: Checking API keys...${NC}"

# Check if secrets exist
GEMINI_EXISTS=$(gcloud secrets list --filter="name:gemini-api-key" --format="value(name)" 2>/dev/null || echo "")
SEARCH_KEY_EXISTS=$(gcloud secrets list --filter="name:google-search-api-key" --format="value(name)" 2>/dev/null || echo "")
SEARCH_ID_EXISTS=$(gcloud secrets list --filter="name:google-search-engine-id" --format="value(name)" 2>/dev/null || echo "")

if [ -z "$GEMINI_EXISTS" ]; then
    echo -e "${YELLOW}⚠ Gemini API key not found in Secret Manager${NC}"
    read -p "Enter Gemini API key (get from https://aistudio.google.com/app/apikey): " GEMINI_KEY
    if [ -n "$GEMINI_KEY" ]; then
        echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=- --replication-policy="automatic"
        echo -e "${GREEN}✓ Gemini API key stored${NC}"
    fi
else
    echo -e "${GREEN}✓ Gemini API key found${NC}"
fi

if [ -z "$SEARCH_KEY_EXISTS" ]; then
    echo -e "${YELLOW}⚠ Google Search API key not found${NC}"
    read -p "Enter Google Search API key (or press Enter to skip): " SEARCH_KEY
    if [ -n "$SEARCH_KEY" ]; then
        echo -n "$SEARCH_KEY" | gcloud secrets create google-search-api-key --data-file=- --replication-policy="automatic"
        echo -e "${GREEN}✓ Google Search API key stored${NC}"
    fi
else
    echo -e "${GREEN}✓ Google Search API key found${NC}"
fi

if [ -z "$SEARCH_ID_EXISTS" ]; then
    echo -e "${YELLOW}⚠ Search Engine ID not found${NC}"
    read -p "Enter Google Search Engine ID (or press Enter to skip): " SEARCH_ID
    if [ -n "$SEARCH_ID" ]; then
        echo -n "$SEARCH_ID" | gcloud secrets create google-search-engine-id --data-file=- --replication-policy="automatic"
        echo -e "${GREEN}✓ Search Engine ID stored${NC}"
    fi
else
    echo -e "${GREEN}✓ Search Engine ID found${NC}"
fi

# Create service account
echo -e "\n${GREEN}Step 4/7: Setting up service account...${NC}"
SA_EMAIL="satyacheck-api@${PROJECT_ID}.iam.gserviceaccount.com"
SA_EXISTS=$(gcloud iam service-accounts list --filter="email:$SA_EMAIL" --format="value(email)" 2>/dev/null || echo "")

if [ -z "$SA_EXISTS" ]; then
    echo -e "${YELLOW}Creating service account...${NC}"
    gcloud iam service-accounts create satyacheck-api --display-name="SatyaCheck API Service Account" || true

    # Grant permissions
    echo -e "${YELLOW}Granting permissions...${NC}"
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/aiplatform.user" --quiet || true
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/cloudtranslate.user" --quiet || true
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/vision.admin" --quiet || true
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/speech.admin" --quiet || true
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/datastore.user" --quiet || true

    echo -e "${GREEN}✓ Service account created and configured${NC}"
else
    echo -e "${GREEN}✓ Service account already exists${NC}"
fi

# Build check
echo -e "\n${GREEN}Step 5/7: Preparing deployment...${NC}"
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
    echo -e "${YELLOW}Please run this script from the project root directory${NC}"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required files found${NC}"

# Deploy
echo -e "\n${GREEN}Step 6/7: Deploying to Cloud Run...${NC}"
echo -e "${YELLOW}This will take 5-10 minutes. Please wait...${NC}"

gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --service-account=$SA_EMAIL \
  --set-env-vars "ENVIRONMENT=production,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,AUTO_ENABLE_DEEPFAKE=false" \
  --quiet

# Get service URL
echo -e "\n${GREEN}Step 7/7: Verifying deployment...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')

# Test health endpoint
echo -e "${YELLOW}Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $SERVICE_URL/health || echo "000")

echo -e "\n${BLUE}"
echo "╔════════════════════════════════════════╗"
echo "║     🎉 Deployment Complete! 🎉        ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "\n${GREEN}✅ Your SatyaCheck AI is live!${NC}"
echo -e "\n${BLUE}Service URL:${NC}"
echo -e "${YELLOW}$SERVICE_URL${NC}"

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo -e "\n${GREEN}✓ Health check: PASSED${NC}"
else
    echo -e "\n${YELLOW}⚠ Health check: $HEALTH_RESPONSE (may take a moment to warm up)${NC}"
fi

echo -e "\n${BLUE}Test your deployment:${NC}"
echo -e "${NC}curl $SERVICE_URL/health${NC}"
echo -e "${NC}curl -X POST $SERVICE_URL/analyze -F 'claim=The Earth is round' -F 'language=en'${NC}"

echo -e "\n${BLUE}Next Steps:${NC}"
echo "1. Update your frontend index.html with the service URL:"
echo -e "   ${YELLOW}const API_URL = '$SERVICE_URL'${NC}"
echo ""
echo "2. View logs:"
echo -e "   ${NC}gcloud run services logs tail $SERVICE_NAME --region $REGION${NC}"
echo ""
echo "3. Monitor your service:"
echo -e "   ${NC}https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME${NC}"
echo ""
echo "4. Set up custom domain (optional):"
echo -e "   ${NC}gcloud run domain-mappings create --service $SERVICE_NAME --domain api.yourdomain.com --region $REGION${NC}"

echo -e "\n${GREEN}Cost Optimization:${NC}"
echo "• Your service scales to zero when idle (no cost)"
echo "• Free tier: 2M requests/month"
echo "• Auto-scales up to 10 instances max"

echo -e "\n${YELLOW}Save these details:${NC}"
echo "Project ID:   $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region:       $REGION"
echo "Service URL:  $SERVICE_URL"

echo -e "\n${GREEN}Deployment successful! 🚀${NC}\n"
