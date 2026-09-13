# Deployment Guide

This guide explains how to deploy the **Enterprise Customer Churn MLOps Platform** to a production environment. 

The project is fully containerized using Docker, allowing you to easily host it on any cloud provider (AWS, DigitalOcean, Google Cloud, Azure) or Platform-as-a-Service (PaaS) like Render or Heroku.

---

## 🏗️ Prerequisites

Before you begin, ensure you have the following installed on your deployment server:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

*(Note: If you are using a PaaS like Render, you do not need to install these manually, as the platform handles containerization natively.)*

---

## 🚀 Option 1: Deploying via Docker Compose (VPS / EC2 / DigitalOcean)

This is the recommended approach for deploying to a Virtual Private Server (VPS). 

1. **Clone the Repository to your Server:**
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd customer-churn-mlops
   ```

2. **Ensure ML Artifacts are Built:**
   If your repository does not include the trained model artifacts, you must generate them first. Run the training script locally or on the server:
   ```bash
   python run_app.py train
   ```

3. **Spin up the Services:**
   Use Docker Compose to build and start the containers in detached mode:
   ```bash
   docker-compose up --build -d
   ```

4. **Verify Deployment:**
   - **Dashboard**: Open `http://<SERVER_IP>:8501` in your browser.
   - **FastAPI Documentation**: Open `http://<SERVER_IP>:8000/docs`.

5. **Stopping the Services:**
   ```bash
   docker-compose down
   ```

---

## ☁️ Option 2: Deploying to Render (PaaS)

[Render](https://render.com/) is an excellent platform for hosting Dockerized applications. You can deploy both the Dashboard and the API as separate Web Services.

### Deploying the FastAPI Service
1. In your Render Dashboard, click **New +** and select **Web Service**.
2. Connect your GitHub repository.
3. **Environment**: Select `Docker`.
4. **Dockerfile Path**: Enter `Dockerfile.api`.
5. **Start Command**: Leave blank (it uses the Dockerfile's CMD).
6. Click **Create Web Service**. 
7. *Note the generated URL for the API (e.g., `https://churn-api.onrender.com`).*

### Deploying the Streamlit Dashboard
1. Create another **Web Service** on Render.
2. Connect the same repository.
3. **Environment**: Select `Docker`.
4. **Dockerfile Path**: Enter `Dockerfile.dashboard`.
5. Under **Advanced**, add an Environment Variable: 
   - `Key`: `API_URL`
   - `Value`: `<YOUR_RENDER_API_URL>` (from the step above).
6. Click **Create Web Service**.

---

## 🔧 Continuous Integration / Continuous Deployment (CI/CD)

Because this project includes `Dockerfile`s, you can easily integrate it into GitHub Actions. A typical CI/CD pipeline would:
1. Trigger on a push to the `main` branch.
2. Run data validation and basic unit tests.
3. Re-train the models (`python run_app.py train`) to generate fresh artifacts.
4. Build the Docker images.
5. Push the images to a container registry (like Docker Hub or AWS ECR).
6. Trigger a deployment hook to restart your cloud instances with the latest image.
