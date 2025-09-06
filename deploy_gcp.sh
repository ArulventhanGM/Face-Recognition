#!/bin/bash

# Google Cloud Platform Deployment Script
# Face Recognition Application

set -e

echo "🚀 Starting Google Cloud Platform Deployment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud SDK is not installed"
    print_status "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_success "Google Cloud SDK found"

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "Not authenticated with Google Cloud"
    print_status "Running authentication..."
    gcloud auth login
fi

print_success "Authenticated with Google Cloud"

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    print_error "No project set. Please set a project first:"
    print_status "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_success "Using project: $PROJECT_ID"

# Enable required APIs
print_status "Enabling required APIs..."
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

print_success "APIs enabled"

# Create App Engine app if it doesn't exist
if ! gcloud app describe &> /dev/null; then
    print_status "Creating App Engine application..."
    gcloud app create --region=us-central1
    print_success "App Engine application created"
else
    print_success "App Engine application already exists"
fi

# Clean up any unwanted files
print_status "Cleaning up unwanted files..."
rm -f debug_*.py test_*.py fix_*.py verify_*.py
rm -rf __pycache__ .pytest_cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

print_success "Cleanup completed"

# Deploy the application
print_status "Deploying to Google App Engine..."
gcloud app deploy app.yaml --quiet

if [ $? -eq 0 ]; then
    print_success "Deployment successful!"
    
    # Get the application URL
    APP_URL=$(gcloud app browse --no-launch-browser 2>&1 | grep -o 'https://[^[:space:]]*')
    
    print_success "Application deployed successfully!"
    print_status "Application URL: $APP_URL"
    print_status "Opening application in browser..."
    
    # Open in browser (optional)
    gcloud app browse
    
else
    print_error "Deployment failed!"
    exit 1
fi

print_success "🎉 Deployment completed successfully!"
print_status "You can view logs with: gcloud app logs tail -s default"
print_status "You can view the app with: gcloud app browse"
