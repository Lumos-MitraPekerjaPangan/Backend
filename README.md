# ML Forecast API for Google Cloud Run

This repository contains a Flask API for machine learning forecasts, configured for deployment to Google Cloud Run with GitHub integration.

## Setup Instructions for GitHub to GCP Deployment

1. Create a Google Cloud Platform project and enable required APIs:
   - Cloud Run API
   - Cloud Build API
   - Secret Manager API
   - Container Registry API

2. Store secrets in Secret Manager:
   - Create a secret `firebase-credentials` with Firebase credentials JSON
   - Create a secret `api-auth-token` with the authentication token

3. Set up Workload Identity Federation (recommended method):
   - Follow [GitHub's documentation](https://github.com/google-github-actions/auth) to set up Workload Identity Federation
   - Create two repository secrets in GitHub:
     - `WIF_PROVIDER`: Your Workload Identity Provider
     - `WIF_SERVICE_ACCOUNT`: Your service account email

4. Alternative: Service Account Key method:
   - Create a service account with the necessary permissions
   - Create a service account key
   - Store the key content as a GitHub secret named `GCP_SA_KEY`
   - Update the GitHub workflow file to use this method

5. Update values in workflow files:
   - In `.github/workflows/cloud-run-deploy.yml` update:
     - `PROJECT_ID`: Your GCP project ID
     - `SERVICE_NAME`: Name for your Cloud Run service (optional)
     - `REGION`: Your preferred GCP region (optional)

## Local Development

To run the API locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` file with required variables
3. Run with: `python api.py`

## API Endpoints

- `/health`: Check API status
- `/forecast/<data_type>`: Get forecast for specific data type
- `/forecast/custom/<data_type>`: Get custom forecast with specified parameters
- `/update-forecast/<data_type>`: Update forecast for specific data type
- `/update-all-models`: Update all models with latest data
