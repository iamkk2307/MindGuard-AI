# MindGuard AI (UDGAM Project)

## 🧠 Mental Health Risk Detection System
A comprehensive AI-powered application designed to detect early signs of mental health risks (Stress, Anxiety, Depression) from text input.

## Features
- **Real-time Analysis**: Instant risk classification.
- **Explainable AI**: Highlights keywords influencing the prediction.
- **Privacy First**: All processing runs within the instance.
- **Resource Hub**: Direct access to emergency helplines.

## 🚀 Quick Start (Local)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```
   *Note: The app will automatically generate synthetic data and train the model on the first run.*

## 🐳 Docker Deployment

1. **Build Image**
   ```bash
   docker build -t mindguard .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 mindguard
   ```

## ☁️ Cloud Deployment (Heroku/Railway)
This project includes a `Procfile` and is ready for PaaS deployment.
1. Connect your repository to the service.
2. The service will detect the Procfile and build using `requirements.txt`.
3. Set environment variables if necessary (none required by default).

## Project Structure
- `app.py`: Main Streamlit application.
- `src/`: Core logic (Data generation, Model training, Preprocessing).
- `models/`: Stores trained model artifacts (auto-generated).
- `data/`: Stores datasets (auto-generated).
