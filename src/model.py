"""
MindGuard AI - Mental Health Risk Detection Model
Uses mental_health_combined_test.csv + custom samples
Logistic Regression (baseline) + XGBoost (improved)
"""
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.preprocess import clean_text

np.random.seed(42)

# Paths
DATA_PATH = 'data/mental_health_combined_test.csv'
MODEL_PATH = 'models/mental_health_model.pkl'
BASELINE_MODEL_PATH = 'models/baseline_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
COMPARISON_PATH = 'models/model_comparison.pkl'


def get_additional_samples():
    """Additional samples to improve coverage for all categories."""
    samples = []
    
    # Normal (positive/neutral)
    normal_texts = [
        "I am feeling relaxed and happy today",
        "Everything is going well in my life",
        "I feel content and peaceful",
        "Life is good and I'm grateful",
        "I'm in a great mood today",
        "Feeling positive and energetic",
        "I had a wonderful day",
        "Things are going smoothly",
        "I feel balanced and calm",
        "I'm happy with my life",
        "Enjoying time with friends and family",
        "I feel healthy and strong",
        "Today was a productive and good day",
        "I'm satisfied with how things are going",
        "Feeling blessed and thankful for everything",
        "I love spending time with my loved ones",
        "Everything feels perfect right now",
        "I'm excited about the future",
    ]
    for t in normal_texts:
        samples.append({'text': t, 'status': 'Normal'})
    
    # Stress (work/deadline pressure)
    stress_texts = [
        "I have so much work to do and not enough time",
        "Deadline pressure is getting to me",
        "I'm overwhelmed with responsibilities",
        "Work is really stressing me out",
        "Too many deadlines to handle",
        "I'm burnt out from work",
        "The workload is crushing me",
        "I'm stressed about exams and grades",
        "So much pressure from all sides",
        "I can't keep up with everything",
        "Work-life balance doesn't exist for me",
        "I'm exhausted from overworking",
        "The pressure at work is unbearable",
        "Finals week is killing me",
        "I'm stressed about money and bills",
    ]
    for t in stress_texts:
        samples.append({'text': t, 'status': 'Stress'})
    
    # Depression - ADDED MORE SAMPLES
    depression_texts = [
        "I am feeling depressed",
        "I feel depressed and sad",
        "I'm feeling really depressed lately",
        "Depression is taking over my life",
        "I feel hopeless and depressed",
        "Everything feels meaningless and empty",
        "I don't see the point in anything anymore",
        "I feel like a burden to everyone",
        "I have no motivation to do anything",
        "I cry all the time for no reason",
        "I feel worthless and useless",
        "Nothing makes me happy anymore",
        "I feel empty inside",
        "I don't want to get out of bed",
        "Life feels pointless",
        "I feel so alone even when surrounded by people",
        "I have lost interest in everything I used to enjoy",
        "I feel numb and disconnected",
        "I can't remember the last time I felt happy",
        "I feel like giving up on everything",
        "I'm feeling bored and depressed",
        "I feel sad and depressed all the time",
        "Depression is ruining my life",
        "I feel so down and depressed",
        "My depression is getting worse",
        "I hate myself and my life",
        "I feel like a failure",
        "Nothing will ever get better",
        "I'm tired of feeling this way",
        "I want the pain to stop",
    ]
    for t in depression_texts:
        samples.append({'text': t, 'status': 'Depression'})
    
    # Anxiety - ADDED MORE SAMPLES
    anxiety_texts = [
        "I can't stop worrying about everything",
        "My heart races and I feel panicked",
        "I have constant anxiety attacks",
        "I'm always nervous and on edge",
        "I feel like something bad is going to happen",
        "I can't relax, my mind won't stop racing",
        "I'm terrified of social situations",
        "Anxiety is controlling my life",
        "I feel anxious all the time",
        "I'm having panic attacks frequently",
        "I can't sleep because of worry",
        "I'm scared of everything",
        "My anxiety is overwhelming",
        "I feel paralyzed by fear",
        "I can't breathe when I'm anxious",
        "I'm constantly worried about the future",
        "I have severe anxiety",
        "I feel like I'm losing control",
        "My anxiety is crippling",
        "I'm afraid to leave my house",
    ]
    for t in anxiety_texts:
        samples.append({'text': t, 'status': 'Anxiety'})
    
    return pd.DataFrame(samples)


def load_and_prepare_data():
    """Loads dataset and balances classes."""
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Add additional samples
    extra = get_additional_samples()
    df = pd.concat([df, extra], ignore_index=True)
    
    # Map labels to 4 categories
    label_map = {
        'Normal': 'Normal',
        'Stress': 'Stress',
        'Anxiety': 'Anxiety',
        'Depression': 'Depression',
        'Suicidal': 'Depression',
    }
    
    df['label'] = df['status'].map(label_map)
    df = df.dropna(subset=['label', 'text'])
    df = df[df['text'].str.len() > 5]
    
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def train_model():
    """Trains Logistic Regression + XGBoost models."""
    print("=" * 60)
    print("MindGuard AI - Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading dataset...")
    df = load_and_prepare_data()
    
    # Preprocess
    print("\n[2/6] Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    
    # TF-IDF
    print("\n[3/6] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    X = vectorizer.fit_transform(df['cleaned_text'])
    print(f"Feature matrix: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    print(f"Classes: {label_encoder.classes_}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # ===== BASELINE: Logistic Regression =====
    print("\n[4/6] Training Logistic Regression (Baseline)...")
    baseline = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    baseline.fit(X_train, y_train)
    
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Baseline F1-Score: {baseline_f1:.4f}")
    
    # ===== IMPROVED: XGBoost =====
    print("\n[5/6] Training XGBoost (Improved)...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"XGBoost F1-Score: {xgb_f1:.4f}")
    
    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (XGBoost)")
    print("=" * 60)
    print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))
    
    # Comparison data
    comparison = {
        'baseline': {
            'name': 'Logistic Regression',
            'accuracy': round(baseline_acc, 4),
            'f1_score': round(baseline_f1, 4),
            'confusion_matrix': confusion_matrix(y_test, baseline_pred).tolist()
        },
        'improved': {
            'name': 'XGBoost',
            'accuracy': round(xgb_acc, 4),
            'f1_score': round(xgb_f1, 4),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist()
        },
        'improvement': {
            'accuracy_gain': round((xgb_acc - baseline_acc) * 100, 2)
        },
        'classes': label_encoder.classes_.tolist()
    }
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Baseline':<15} {'XGBoost':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {baseline_acc*100:.2f}%{'':<8} {xgb_acc*100:.2f}%{'':<8} {(xgb_acc-baseline_acc)*100:+.2f}%")
    print(f"{'F1-Score':<15} {baseline_f1*100:.2f}%{'':<8} {xgb_f1*100:.2f}%{'':<8} {(xgb_f1-baseline_f1)*100:+.2f}%")
    
    # Save
    print("\n[6/6] Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb, MODEL_PATH)
    joblib.dump(baseline, BASELINE_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(comparison, COMPARISON_PATH)
    
    print("✅ Training complete! Models saved to models/")
    return comparison


def get_model_comparison():
    try:
        return joblib.load(COMPARISON_PATH)
    except:
        return None


def predict_risk(text, use_baseline=False):
    """Predicts mental health risk from any text input."""
    try:
        model = joblib.load(BASELINE_MODEL_PATH if use_baseline else MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        return {"error": "Model not trained yet."}
    
    # Clean and vectorize
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    # Predict
    pred_idx = model.predict(vec)[0]
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(vec)[0]
    
    # Risk mapping
    risk_map = {
        'Normal': 'Low',
        'Stress': 'Medium', 
        'Anxiety': 'High',
        'Depression': 'High'
    }
    
    # XAI: Get influential words
    features = vectorizer.get_feature_names_out()
    indices = vec.nonzero()[1]
    
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        words = [(features[i], imp[i]) for i in indices if imp[i] > 0]
    else:
        coefs = model.coef_[pred_idx]
        words = [(features[i], coefs[i]) for i in indices if coefs[i] > 0]
    
    words.sort(key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in words[:5]]
    
    # Class probabilities
    class_probs = {
        label_encoder.classes_[i]: round(float(p) * 100, 2) 
        for i, p in enumerate(probs)
    }
    
    return {
        "prediction": prediction,
        "confidence": round(float(max(probs)) * 100, 2),
        "risk_level": risk_map.get(prediction, 'Unknown'),
        "highlighted_words": keywords,
        "explanation_text": f"Text analysis indicates {prediction.lower()} patterns.",
        "class_probabilities": class_probs,
        "model_used": "Baseline (Logistic Regression)" if use_baseline else "XGBoost"
    }


if __name__ == "__main__":
    train_model()
