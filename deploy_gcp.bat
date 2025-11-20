@echo off
REM Google Cloud Platform Deployment Script for Windows
REM Face Recognition Application

echo Starting Google Cloud Platform Deployment...

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Google Cloud SDK is not installed
    echo Please install it from: https://cloud.google.com/sdk/docs/install
    pause
    exit /b 1
)

echo SUCCESS: Google Cloud SDK found

REM Check authentication
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr . >nul
if errorlevel 1 (
    echo WARNING: Not authenticated with Google Cloud
    echo Running authentication...
    gcloud auth login
)

echo SUCCESS: Authenticated with Google Cloud

REM Get project ID
for /f "delims=" %%i in ('gcloud config get-value project') do set PROJECT_ID=%%i
if "%PROJECT_ID%"=="" (
    echo ERROR: No project set. Please set a project first:
    echo gcloud config set project YOUR_PROJECT_ID
    pause
    exit /b 1
)

echo SUCCESS: Using project: %PROJECT_ID%

REM Enable required APIs
echo INFO: Enabling required APIs...
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

echo SUCCESS: APIs enabled

REM Check if App Engine app exists
gcloud app describe >nul 2>&1
if errorlevel 1 (
    echo INFO: Creating App Engine application...
    gcloud app create --region=us-central1
    echo SUCCESS: App Engine application created
) else (
    echo SUCCESS: App Engine application already exists
)

REM Clean up unwanted files
echo INFO: Cleaning up unwanted files...
del /q debug_*.py test_*.py fix_*.py verify_*.py 2>nul
rmdir /s /q __pycache__ 2>nul
rmdir /s /q .pytest_cache 2>nul

echo SUCCESS: Cleanup completed

REM Deploy the application
echo INFO: Deploying to Google App Engine...
gcloud app deploy app.yaml --quiet

if errorlevel 1 (
    echo ERROR: Deployment failed!
    pause
    exit /b 1
)

echo SUCCESS: Deployment successful!

REM Get the application URL and open browser
echo INFO: Opening application in browser...
gcloud app browse

echo SUCCESS: Deployment completed successfully!
echo You can view logs with: gcloud app logs tail -s default
echo You can view the app with: gcloud app browse

pause
