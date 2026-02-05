"""
MindGuard AI - Mental Health Risk Detection System
Streamlit application with responsive UI, color-coded risk cards, and XAI
Optimized for offline-first, privacy-focused deployment
"""
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from src.model import predict_risk, get_model_comparison

# ========== PAGE CONFIG (Must be first) ==========
st.set_page_config(
    page_title="MindGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SESSION STATE INITIALIZATION ==========
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'voice_text' not in st.session_state:
    st.session_state['voice_text'] = ""

# ========== VOICE INPUT - CHECK FOR RESULT ==========
# This must run at module level to catch the redirect
if 'voice_result' in st.query_params:
    voice_result = st.query_params.get('voice_result', '')
    if voice_result:
        st.session_state['voice_text'] = voice_result
        st.query_params.clear()
        st.rerun()

# ========== VOICE INPUT COMPONENT ==========
def speech_to_text_component():
    """Browser-based speech recognition using popup window to bypass iframe restrictions"""
    
    component_html = '''
    <div id="speech-container" style="text-align: center; padding: 5px;">
        <button id="recordBtn" onclick="openVoicePopup()" style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 15px;
            cursor: pointer;
        ">🎙️ Record Voice</button>
        <span id="status" style="color: #888; margin-left: 10px; font-size: 14px;"></span>
    </div>
    <script>
        function openVoicePopup() {
            const popupHtml = `
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Voice Recording</title>
                    <style>
                        body { font-family: Arial; background: #1a1a2e; color: white; text-align: center; padding: 30px; }
                        button { padding: 15px 30px; font-size: 16px; margin: 10px; border-radius: 25px; cursor: pointer; border: none; }
                        #startBtn { background: #667eea; color: white; }
                        #stopBtn { background: #e53e3e; color: white; }
                        #sendBtn { background: #38ef7d; color: black; display: none; }
                        #result { background: rgba(255,255,255,0.1); padding: 15px; margin: 15px; border-radius: 10px; min-height: 60px; }
                    </style>
                </head>
                <body>
                    <h2>🎙️ Voice Recording</h2>
                    <button id="startBtn" onclick="startRec()">🎙️ Start</button>
                    <button id="stopBtn" onclick="stopRec()" disabled>⏹️ Stop</button>
                    <button id="sendBtn" onclick="sendText()">✅ Use This Text</button>
                    <p id="status">Click Start and speak...</p>
                    <div id="result"></div>
                    <script>
                        let recognition, transcript = '';
                        if ('webkitSpeechRecognition' in window) {
                            recognition = new webkitSpeechRecognition();
                            recognition.continuous = true;
                            recognition.interimResults = true;
                            recognition.lang = 'en-US';
                            recognition.onstart = () => {
                                document.getElementById('status').innerHTML = '🔴 Listening...';
                                document.getElementById('startBtn').disabled = true;
                                document.getElementById('stopBtn').disabled = false;
                                document.getElementById('sendBtn').style.display = 'none';
                            };
                            recognition.onresult = (e) => {
                                transcript = '';
                                for (let i = 0; i < e.results.length; i++) {
                                    transcript += e.results[i][0].transcript;
                                }
                                document.getElementById('result').innerHTML = transcript;
                            };
                            recognition.onerror = (e) => {
                                document.getElementById('status').innerHTML = '❌ Error: ' + e.error;
                                document.getElementById('startBtn').disabled = false;
                                document.getElementById('stopBtn').disabled = true;
                            };
                            recognition.onend = () => {
                                document.getElementById('status').innerHTML = '✅ Done! Click "Use This Text" to continue.';
                                document.getElementById('startBtn').disabled = false;
                                document.getElementById('stopBtn').disabled = true;
                                if (transcript) document.getElementById('sendBtn').style.display = 'inline-block';
                            };
                        }
                        function startRec() { transcript = ''; document.getElementById('result').innerHTML = ''; recognition.start(); }
                        function stopRec() { recognition.stop(); }
                        function sendText() {
                            if (window.opener && window.opener.top) {
                                const url = window.opener.top.location.href.split('?')[0] + '?voice_result=' + encodeURIComponent(transcript);
                                window.opener.top.location.href = url;
                                window.close();
                            }
                        }
                    <\\/script>
                </body>
                </html>
            `;
            const popup = window.open('', 'VoiceRecording', 'width=500,height=400,top=100,left=100');
            popup.document.write(popupHtml);
            popup.document.close();
        }
    </script>
    '''
    return components.html(component_html, height=50)

# ========== AUTO-SETUP FOR PLUG-AND-PLAY DEMO ==========
import os
from src.model import train_model

if not os.path.exists('models/mental_health_model.pkl'):
    with st.spinner("🚀 First-time setup: Training models on real dataset... (1-2 minutes)"):
        train_model()
        st.success("✅ System initialized successfully!")
        st.rerun()

# ========== RESPONSIVE CSS ==========
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: clamp(1rem, 2vw, 1.3rem);
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Risk Cards - Color Coded */
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    .risk-title {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .risk-value {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 800;
        text-transform: uppercase;
    }
    
    .risk-confidence {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    /* Keyword Tags */
    .keyword-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }
    
    .keyword-tag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: linear-gradient(135deg, #fff5f5, #ffe6e6);
        border-left: 5px solid #e53e3e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .disclaimer-box h4 {
        color: #c53030;
        margin: 0 0 0.5rem 0;
    }
    
    .disclaimer-box p {
        color: #742a2a;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .comparison-table th, .comparison-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .comparison-table tr:hover {
        background-color: #f5f5f5;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .stApp { padding: 1rem; }
        .risk-card { padding: 1.5rem; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🧠 MindGuard AI")
    st.caption("AI-Powered Mental Health Risk Detection")
    
    st.markdown("---")
    
    # Model Info
    st.subheader("⚙️ Model Information")
    comparison = get_model_comparison()
    if comparison:
        st.write(f"**Primary:** XGBoost")
        st.write(f"**Accuracy:** {comparison['improved']['accuracy']*100:.1f}%")
        st.write(f"**F1-Score:** {comparison['improved']['f1_score']*100:.1f}%")
    
    st.markdown("---")
    
    # Ethical Disclaimer
    st.subheader("⚠️ Important Notice")
    st.error("""
    **This is NOT a medical diagnosis tool.**
    
    MindGuard AI is designed for awareness and early screening only. Always consult qualified mental health professionals for proper diagnosis and treatment.
    """)
    
    st.markdown("---")
    st.caption("🔒 Privacy First: All data processed locally")
    st.caption("Developed for UDGAM Project | Woxsen University")

# ========== MAIN CONTENT ==========
st.markdown('<div class="main-header">🧠 MindGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mental Health Risk Detection System</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "📊 Model Comparison", "📜 History", "ℹ️ About"])

# ========== TAB 1: ANALYZE ==========
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### How are you feeling today?")
        
        # Voice input using Web Speech API
        st.markdown("**🎙️ Voice Input (Chrome recommended):**")
        speech_to_text_component()
        
        # Text area
        text_input = st.text_area(
            "Share your thoughts",
            value=st.session_state['voice_text'],
            placeholder="Type or paste your transcribed speech here...",
            height=150,
            label_visibility="collapsed",
            key="text_input_area"
        )
        
        # Update session state if user types manually
        if text_input != st.session_state['voice_text']:
            st.session_state['voice_text'] = text_input
        
        analyze_btn = st.button("🔍 Analyze My Text", type="primary", use_container_width=True)
    
    if analyze_btn and text_input:
        with st.spinner("Analyzing patterns..."):
            result = predict_risk(text_input)
        
        if "error" in result:
            st.error(result['error'])
        else:
            # Save to history
            st.session_state['history'].append({
                "Text": text_input[:100] + "..." if len(text_input) > 100 else text_input,
                "Prediction": result['prediction'],
                "Risk Level": result['risk_level'],
                "Confidence": f"{result['confidence']}%"
            })
            
            st.markdown("---")
            
            # Risk Level Card
            risk = result['risk_level']
            risk_class = "risk-low" if risk == "Low" else "risk-medium" if risk == "Medium" else "risk-high"
            
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <div class="risk-title">Detected Risk Level</div>
                <div class="risk-value">{risk}</div>
                <div class="risk-confidence">Prediction: {result['prediction']} | Confidence: {result['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Two columns for details
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 📊 Probability Distribution")
                probs = result['class_probabilities']
                prob_df = pd.DataFrame({
                    'Category': list(probs.keys()),
                    'Probability (%)': list(probs.values())
                })
                st.bar_chart(prob_df.set_index('Category'))
            
            with col_right:
                st.markdown("### 🔍 Explainable AI Insights")
                st.info(result['explanation_text'])
                
                if result['highlighted_words']:
                    st.markdown("**Influential Keywords:**")
                    keywords_html = ''.join([
                        f'<span class="keyword-tag">{word}</span>' 
                        for word in result['highlighted_words']
                    ])
                    st.markdown(f'<div class="keyword-container">{keywords_html}</div>', unsafe_allow_html=True)
                else:
                    st.write("Overall context influenced the prediction, no single dominant keyword.")
            
            # High Risk Warning
            if risk == "High":
                st.markdown("""
                <div class="disclaimer-box">
                    <h4>⚠️ Important</h4>
                    <p>High risk detected. Please consider reaching out to a mental health professional or calling a helpline listed in the sidebar. Remember, seeking help is a sign of strength.</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif analyze_btn and not text_input:
        st.warning("Please enter some text to analyze.")

# ========== TAB 2: MODEL COMPARISON ==========
with tab2:
    st.markdown("### 📊 Baseline vs Improved Model Comparison")
    
    comparison = get_model_comparison()
    
    if comparison:
        # Comparison Table
        baseline_acc = comparison['baseline']['accuracy'] * 100
        improved_acc = comparison['improved']['accuracy'] * 100
        baseline_f1 = comparison['baseline']['f1_score'] * 100
        improved_f1 = comparison['improved']['f1_score'] * 100
        acc_gain = comparison['improvement'].get('accuracy_gain', improved_acc - baseline_acc)
        f1_gain = improved_f1 - baseline_f1
        
        st.markdown(f"""
        | Metric | Baseline (Logistic Regression) | Improved (XGBoost) | Improvement |
        |--------|--------------------------------|-------------------|-------------|
        | **Accuracy** | {baseline_acc:.2f}% | {improved_acc:.2f}% | {acc_gain:+.2f}% |
        | **F1-Score** | {baseline_f1:.2f}% | {improved_f1:.2f}% | {f1_gain:+.2f}% |
        """)
        
        st.markdown("---")
        
        # Confusion Matrices
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Baseline Confusion Matrix")
            cm_baseline = pd.DataFrame(
                comparison['baseline']['confusion_matrix'],
                index=comparison['classes'],
                columns=comparison['classes']
            )
            st.dataframe(cm_baseline, use_container_width=True)
        
        with col2:
            st.markdown("#### XGBoost Confusion Matrix")
            cm_improved = pd.DataFrame(
                comparison['improved']['confusion_matrix'],
                index=comparison['classes'],
                columns=comparison['classes']
            )
            st.dataframe(cm_improved, use_container_width=True)
        
        st.success(f"✅ XGBoost shows **{comparison['improvement']['accuracy_gain']:+.2f}%** accuracy improvement over baseline!")
    else:
        st.info("Model comparison data will be available after training completes.")

# ========== TAB 3: HISTORY ==========
with tab3:
    st.markdown("### 📜 Session Analysis History")
    st.caption("Predictions are stored locally in session state for privacy.")
    
    if st.session_state['history']:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df, use_container_width=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state['history'] = []
                st.rerun()
    else:
        st.info("No analysis performed in this session yet. Go to the Analyze tab to start.")

# ========== TAB 4: ABOUT ==========
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ℹ️ About MindGuard AI")
        st.markdown("""
        MindGuard AI is an AI-powered mental health awareness tool that uses Natural Language Processing (NLP) 
        to analyze text and detect potential mental health risk indicators.
        
        **How it works:**
        1. **Text Input:** Share your thoughts in natural language
        2. **NLP Processing:** Text is cleaned and converted to features
        3. **ML Classification:** XGBoost model predicts risk category
        4. **Explainable AI:** Key influential words are highlighted
        
        **Categories:**
        - 😊 **Normal** - Low Risk
        - 😰 **Stress** - Medium Risk  
        - 😟 **Anxiety** - High Risk
        - 😔 **Depression** - High Risk
        
        **Technology Stack:**
        - Python, Streamlit, XGBoost, Scikit-learn
        - TF-IDF Vectorization, NLTK
        - Docker, Cloud Deployment Ready
        """)
    
    with col2:
        st.markdown("### ⚠️ Ethical Statement")
        st.markdown("""
        <div class="disclaimer-box">
            <h4>⚠️ Medical Disclaimer</h4>
            <p>
            MindGuard AI is <strong>NOT a medical diagnosis tool</strong>. It is designed for 
            educational and awareness purposes only.
            <br><br>
            <strong>What this tool is:</strong><br>
            ✓ An awareness and early screening aid<br>
            ✓ A demonstration of AI in mental health<br>
            ✓ A privacy-first, offline-capable system<br>
            <br>
            <strong>What this tool is NOT:</strong><br>
            ✗ A replacement for professional diagnosis<br>
            ✗ A substitute for therapy or counseling<br>
            ✗ 100% accurate or definitive<br>
            <br>
            If you are experiencing mental health issues, please reach out to a qualified 
            mental health professional or call helplines listed in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    🧠 MindGuard AI | UDGAM Project | Woxsen University | 2026<br>
    <span style="font-size: 0.75rem;">🔒 Privacy First: All data processed locally | No data stored or transmitted</span>
</div>
""", unsafe_allow_html=True)
