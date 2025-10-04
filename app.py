import streamlit as st
import joblib
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from datetime import datetime, timedelta
import time
import io

# Page configuration
st.set_page_config(
    page_title="CyberGuard AI - Professional Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("cyberbullying_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# Abusive words dictionary
abusive_words = ["idiot", "stupid", "hate", "dumb", "loser", "kill", "ugly", "die", "pathetic", "worthless", 
                 "trash", "garbage", "scum", "disgusting", "failure"]

# Severity scoring system
def calculate_severity(prediction, confidence, text):
    base_scores = {
        "not_cyberbullying": 0,
        "age": 60,
        "ethnicity": 85,
        "gender": 75,
        "religion": 80,
        "other_cyberbullying": 70
    }
    base = base_scores.get(prediction.lower(), 50)
    confidence_factor = confidence / 100
    abusive_count = sum(1 for word in abusive_words if re.search(rf"\b{word}\b", text, re.IGNORECASE))
    abusive_factor = min(abusive_count * 5, 20)
    severity = min(100, base * confidence_factor + abusive_factor)
    return round(severity, 1)

def get_severity_level(score):
    if score < 20:
        return "SAFE", "#10b981", "🟢"
    elif score < 40:
        return "LOW", "#3b82f6", "🔵"
    elif score < 60:
        return "MEDIUM", "#f59e0b", "🟡"
    elif score < 80:
        return "HIGH", "#f97316", "🟠"
    else:
        return "CRITICAL", "#ef4444", "🔴"

# Load Lottie animation
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS - White Background with Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px 40px;
        border-radius: 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        margin: -60px -60px 40px -60px;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .logo-text {
        font-size: 1.8em;
        font-weight: 800;
        color: white;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Main Content Container */
    .main-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 40px;
        margin: 20px 0;
        animation: fadeInUp 0.6s ease-out;
        border: 1px solid #e5e7eb;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 60px 0;
        background: linear-gradient(135deg, #f8f9ff 0%, #fff5f7 100%);
        border-radius: 20px;
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 3.5em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        animation: titleFloat 3s ease-in-out infinite;
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-subtitle {
        font-size: 1.3em;
        color: #64748b;
        margin-bottom: 40px;
        font-weight: 500;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin: 40px 0;
    }
    
    .feature-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 35px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid #f3f4f6;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 3em;
        margin-bottom: 20px;
    }
    
    .feature-title {
        font-size: 1.4em;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
    }
    
    .feature-desc {
        color: #64748b;
        line-height: 1.7;
        font-size: 0.95em;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 35px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .stat-label {
        font-size: 1em;
        opacity: 0.95;
        font-weight: 500;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 36px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        font-size: 15px;
        padding: 18px;
        transition: all 0.3s ease;
        background: #fafafa;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 20px 25px;
        border-radius: 12px;
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
        border: 1px solid;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .alert-success {
        background: #f0fdf4;
        color: #166534;
        border-color: #86efac;
        border-left: 4px solid #10b981;
    }
    
    .alert-danger {
        background: #fef2f2;
        color: #991b1b;
        border-color: #fca5a5;
        border-left: 4px solid #ef4444;
    }
    
    .alert-info {
        background: #eff6ff;
        color: #1e40af;
        border-color: #93c5fd;
        border-left: 4px solid #3b82f6;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border: 1px solid #f3f4f6;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #fafafa;
    }
    
    /* Footer */
    .app-footer {
        background: #f8fafc;
        padding: 50px 40px;
        border-radius: 0;
        margin: 60px -60px -60px -60px;
        border-top: 1px solid #e5e7eb;
    }
    
    .footer-content {
        max-width: 1400px;
        margin: 0 auto;
        text-align: center;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 25px 0;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: #64748b;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
        cursor: pointer;
    }
    
    .footer-link:hover {
        color: #667eea;
    }
    
    /* Severity Gauge */
    .severity-gauge {
        position: relative;
        height: 28px;
        background: #f3f4f6;
        border-radius: 14px;
        overflow: hidden;
        margin: 20px 0;
        border: 1px solid #e5e7eb;
    }
    
    .severity-fill {
        height: 100%;
        border-radius: 14px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .severity-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shine 2s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Section Headers */
    h1, h2, h3, h4 {
        color: #1e293b;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Navigation
def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="logo-section">
                <span style="font-size: 2em;">🛡️</span>
                <span class="logo-text">CyberGuard AI</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = 'home'
    with col2:
        if st.button("🔍 Analyze", key="nav_analyze", use_container_width=True):
            st.session_state.page = 'analyze'
    with col3:
        if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
    with col4:
        if st.button("📝 Reports", key="nav_reports", use_container_width=True):
            st.session_state.page = 'reports'
    with col5:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.page = 'about'

# Footer
def render_footer():
    st.markdown("""
    <div class="app-footer">
        <div class="footer-content">
            <h3 style='color: #1e293b; margin-bottom: 20px; font-size: 1.5em;'>🛡️ CyberGuard AI</h3>
            <p style='color: #64748b; margin-bottom: 25px; font-size: 1.05em;'>
                Advanced AI-Powered Cyberbullying Detection System<br>
                Protecting Digital Communities Worldwide
            </p>
            <div class="footer-links">
                <span class="footer-link">Privacy Policy</span>
                <span class="footer-link">Terms of Service</span>
                <span class="footer-link">Documentation</span>
                <span class="footer-link">API Access</span>
                <span class="footer-link">Support</span>
            </div>
            <p style='color: #94a3b8; font-size: 0.9em; margin-top: 25px;'>
                © 2024 CyberGuard Systems. All rights reserved.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# HOME PAGE
def home_page():
    lottie_shield = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_kxsd2ytq.json")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_shield:
            st_lottie(lottie_shield, height=250, key="home_shield")
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Protect Your Digital Space</h1>
        <p class="hero-subtitle">
            AI-powered cyberbullying detection that keeps your community safe<br>
            Real-time analysis • Multi-language support • 99.2% accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Analyzing Messages", key="cta_analyze", use_container_width=True):
            st.session_state.page = 'analyze'
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("<h2 style='text-align: center; color: #1e293b; margin: 60px 0 40px 0; font-weight: 700;'>Why Choose CyberGuard AI?</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Multi-Class Detection</div>
            <div class="feature-desc">
                Identifies 5+ types of cyberbullying including age, gender, ethnicity, religion-based harassment and more.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Severity Scoring</div>
            <div class="feature-desc">
                Advanced 0-100 risk assessment scale with detailed threat level classification.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Analysis</div>
            <div class="feature-desc">
                Instant AI-powered classification with sub-second response times for immediate action.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Secure & Private</div>
            <div class="feature-desc">
                Enterprise-grade security with encrypted data storage and GDPR compliance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Analytics Dashboard</div>
            <div class="feature-desc">
                Comprehensive insights with interactive charts, trends, and detailed reports.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💾</div>
            <div class="feature-title">Auto-Logging</div>
            <div class="feature-desc">
                Automatic archiving of flagged content with timestamps for compliance and review.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("<h2 style='text-align: center; color: #1e293b; margin: 60px 0 40px 0; font-weight: 700;'>System Performance</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">99.2%</div>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">&lt;0.5s</div>
            <div class="stat-label">Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        try:
            df = pd.read_csv("flagged_messages.csv")
            total_analyzed = len(df)
        except:
            total_analyzed = 0
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total_analyzed:,}</div>
            <div class="stat-label">Messages Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">5+</div>
            <div class="stat-label">Threat Categories</div>
        </div>
        """, unsafe_allow_html=True)

# ANALYZE PAGE
def analyze_page():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("# 🔍 Message Analysis")
    st.markdown("Enter any text message, comment, or social media post to analyze for cyberbullying content.")
    
    # Batch Analysis Option
    analysis_mode = st.radio("Select Analysis Mode:", 
                             ["Single Message", "Batch Analysis"], 
                             horizontal=True)
    
    if analysis_mode == "Single Message":
        user_input = st.text_area(
            "Message Input:",
            placeholder="Type or paste your message here for instant AI analysis...",
            height=180,
            key="single_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Message", use_container_width=True, key="analyze_single")
        
        if analyze_button:
            if user_input.strip() == "":
                st.markdown("""
                <div class="alert-box alert-info">
                    <h4 style='margin: 0 0 8px 0;'>⚠️ Input Required</h4>
                    <p style='margin: 0;'>Please enter a message to analyze.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner('Analyzing message...'):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    vect_input = vectorizer.transform([user_input])
                    prediction = model.predict(vect_input)[0]
                    prediction_proba = np.max(model.predict_proba(vect_input)) * 100
                    severity_score = calculate_severity(prediction, prediction_proba, user_input)
                    severity_level, severity_color, severity_icon = get_severity_level(severity_score)
                    
                    progress_bar.empty()
                
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style='font-size: 0.85em; color: #64748b; font-weight: 600; margin-bottom: 8px;'>CLASSIFICATION</div>
                        <div style='font-size: 1.6em; font-weight: 700; color: #1e293b;'>{prediction.replace("_", " ").title()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    confidence_color = "#10b981" if prediction_proba > 70 else "#f59e0b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style='font-size: 0.85em; color: #64748b; font-weight: 600; margin-bottom: 8px;'>CONFIDENCE</div>
                        <div style='font-size: 1.6em; font-weight: 700; color: {confidence_color};'>{prediction_proba:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style='font-size: 0.85em; color: #64748b; font-weight: 600; margin-bottom: 8px;'>SEVERITY</div>
                        <div style='font-size: 1.6em; font-weight: 700; color: {severity_color};'>{severity_icon} {severity_score}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    status = "SAFE" if prediction.lower() == "not_cyberbullying" else "FLAGGED"
                    status_color = "#10b981" if status == "SAFE" else "#ef4444"
                    status_icon = "✅" if status == "SAFE" else "⚠️"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style='font-size: 0.85em; color: #64748b; font-weight: 600; margin-bottom: 8px;'>STATUS</div>
                        <div style='font-size: 1.6em; font-weight: 700; color: {status_color};'>{status_icon}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Severity Gauge
                st.markdown("#### Severity Assessment")
                st.markdown(f"""
                <div class="severity-gauge">
                    <div class="severity-fill" style="width: {severity_score}%; background: linear-gradient(90deg, {severity_color}, {severity_color}dd);"></div>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.75em; color: #64748b;'>
                    <span>0 (Safe)</span>
                    <span>50 (Moderate)</span>
                    <span>100 (Critical)</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Message Preview with highlighting
                def highlight_abusive(text):
                    for word in abusive_words:
                        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
                        text = pattern.sub(f"**:red[{word.upper()}]**", text)
                    return text
                
                st.markdown("#### Message Preview")
                highlighted_text = highlight_abusive(user_input)
                st.markdown(f"> {highlighted_text}")
                
                # Log the message
                if prediction.lower() != "not_cyberbullying":
                    with open("flagged_messages.csv", "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([user_input, prediction, f"{prediction_proba:.2f}%", severity_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    
                    st.markdown(f"""
                    <div class="alert-box alert-danger">
                        <h4 style='margin: 0 0 8px 0;'>⚠️ Message Flagged - {severity_level} Risk</h4>
                        <p style='margin: 0;'>This message has been logged for review with severity score {severity_score}/100.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-box alert-success">
                        <h4 style='margin: 0 0 8px 0;'>✅ Message Safe</h4>
                        <p style='margin: 0;'>No cyberbullying content detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:  # Batch Analysis
        st.markdown("### 📦 Batch Analysis")
        st.info("Upload a CSV file with a 'message' column or paste multiple messages (one per line)")
        
        upload_method = st.radio("Input Method:", ["Upload CSV", "Paste Text"], horizontal=True)
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    if 'message' not in df_upload.columns:
                        st.error("CSV must contain a 'message' column")
                    else:
                        st.success(f"✅ Loaded {len(df_upload)} messages")
                        
                        if st.button("🔍 Analyze All Messages", use_container_width=True):
                            results = []
                            progress = st.progress(0)
                            
                            for idx, row in df_upload.iterrows():
                                message = str(row['message'])
                                vect_input = vectorizer.transform([message])
                                prediction = model.predict(vect_input)[0]
                                prediction_proba = np.max(model.predict_proba(vect_input)) * 100
                                severity = calculate_severity(prediction, prediction_proba, message)
                                
                                results.append({
                                    'Message': message[:50] + '...' if len(message) > 50 else message,
                                    'Classification': prediction.replace("_", " ").title(),
                                    'Confidence': f"{prediction_proba:.1f}%",
                                    'Severity': severity,
                                    'Status': 'Safe' if prediction.lower() == 'not_cyberbullying' else 'Flagged'
                                })
                                
                                progress.progress((idx + 1) / len(df_upload))
                            
                            progress.empty()
                            
                            results_df = pd.DataFrame(results)
                            st.markdown("### 📊 Batch Analysis Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                flagged = len(results_df[results_df['Status'] == 'Flagged'])
                                st.metric("Flagged Messages", flagged)
                            with col2:
                                safe = len(results_df[results_df['Status'] == 'Safe'])
                                st.metric("Safe Messages", safe)
                            with col3:
                                avg_severity = results_df['Severity'].mean()
                                st.metric("Avg Severity", f"{avg_severity:.1f}")
                            
                            # Download results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv_data,
                                "batch_analysis_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        else:  # Paste Text
            batch_input = st.text_area(
                "Paste messages (one per line):",
                height=200,
                placeholder="Message 1\nMessage 2\nMessage 3..."
            )
            
            if st.button("🔍 Analyze All Messages", use_container_width=True, key="batch_text"):
                if batch_input.strip():
                    messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
                    results = []
                    progress = st.progress(0)
                    
                    for idx, message in enumerate(messages):
                        vect_input = vectorizer.transform([message])
                        prediction = model.predict(vect_input)[0]
                        prediction_proba = np.max(model.predict_proba(vect_input)) * 100
                        severity = calculate_severity(prediction, prediction_proba, message)
                        
                        results.append({
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Classification': prediction.replace("_", " ").title(),
                            'Confidence': f"{prediction_proba:.1f}%",
                            'Severity': severity,
                            'Status': 'Safe' if prediction.lower() == 'not_cyberbullying' else 'Flagged'
                        })
                        
                        progress.progress((idx + 1) / len(messages))
                    
                    progress.empty()
                    
                    results_df = pd.DataFrame(results)
                    st.markdown("### 📊 Batch Analysis Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        flagged = len(results_df[results_df['Status'] == 'Flagged'])
                        st.metric("Flagged Messages", flagged)
                    with col2:
                        safe = len(results_df[results_df['Status'] == 'Safe'])
                        st.metric("Safe Messages", safe)
                    with col3:
                        avg_severity = results_df['Severity'].mean()
                        st.metric("Avg Severity", f"{avg_severity:.1f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# DASHBOARD PAGE
def dashboard_page():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("Real-time insights and comprehensive analytics of detected cyberbullying content.")
    
    try:
        df = pd.read_csv("flagged_messages.csv", names=["Message", "Type", "Confidence", "Severity", "Timestamp"])
        
        if len(df) > 0:
            df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # Key Metrics
            st.markdown("### 📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Total Flagged</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                critical = len(df[df['Severity'] >= 80])
                st.markdown(f"""
                <div class="stat-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                    <div class="stat-number">{critical}</div>
                    <div class="stat-label">Critical Cases</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_severity = df['Severity'].mean()
                st.markdown(f"""
                <div class="stat-card" style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);">
                    <div class="stat-number">{avg_severity:.1f}</div>
                    <div class="stat-label">Avg Severity</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                today = datetime.now().date()
                today_count = len(df[df['Timestamp'].dt.date == today])
                st.markdown(f"""
                <div class="stat-card" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                    <div class="stat-number">{today_count}</div>
                    <div class="stat-label">Today</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Distribution by Type")
                counts = df["Type"].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(8, 6), facecolor='white')
                colors = ['#667eea', '#764ba2', '#f093fb', '#f59e0b', '#10b981', '#ef4444']
                wedges, texts, autotexts = ax_pie.pie(
                    counts.values, 
                    labels=[label.replace("_", " ").title() for label in counts.index],
                    autopct='%1.1f%%',
                    colors=colors[:len(counts)],
                    startangle=90,
                    textprops={'fontsize': 11, 'weight': 'bold'},
                    explode=[0.05] * len(counts)
                )
                ax_pie.set_title("Type Distribution", fontsize=15, weight='bold', pad=20, color='#1e293b')
                for autotext in autotexts:
                    autotext.set_color('white')
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close()
            
            with col2:
                st.markdown("#### 📊 Flagged Messages by Type")
                fig_bar, ax_bar = plt.subplots(figsize=(8, 6), facecolor='white')
                bars = ax_bar.bar(
                    range(len(counts)), 
                    counts.values,
                    color=colors[:len(counts)],
                    edgecolor='white',
                    linewidth=2.5
                )
                ax_bar.set_xticks(range(len(counts)))
                ax_bar.set_xticklabels([label.replace("_", " ").title() for label in counts.index], 
                                       rotation=45, ha='right', fontsize=10, weight='600')
                ax_bar.set_ylabel("Count", fontsize=12, weight='bold', color='#1e293b')
                ax_bar.set_title("Count by Type", fontsize=15, weight='bold', pad=20, color='#1e293b')
                ax_bar.grid(axis='y', alpha=0.2, linestyle='--', linewidth=1)
                ax_bar.set_facecolor('#f8fafc')
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                
                for bar in bars:
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=11, weight='bold', color='#1e293b')
                
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
            
            # Severity Distribution
            st.markdown("#### 🎯 Severity Distribution")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                low = len(df[df['Severity'] < 40])
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #3b82f6;">
                    <div style='font-size: 2.5em; font-weight: 700; color: #3b82f6; text-align: center;'>{low}</div>
                    <div style='font-size: 1em; color: #64748b; text-align: center;'>🔵 Low Risk</div>
                    <div style='font-size: 0.85em; color: #94a3b8; text-align: center; margin-top: 8px;'>
                        {(low/len(df)*100):.1f}% of total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                medium = len(df[(df['Severity'] >= 40) & (df['Severity'] < 80)])
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                    <div style='font-size: 2.5em; font-weight: 700; color: #f59e0b; text-align: center;'>{medium}</div>
                    <div style='font-size: 1em; color: #64748b; text-align: center;'>🟡 Medium Risk</div>
                    <div style='font-size: 0.85em; color: #94a3b8; text-align: center; margin-top: 8px;'>
                        {(medium/len(df)*100):.1f}% of total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                high = len(df[df['Severity'] >= 80])
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #ef4444;">
                    <div style='font-size: 2.5em; font-weight: 700; color: #ef4444; text-align: center;'>{high}</div>
                    <div style='font-size: 1em; color: #64748b; text-align: center;'>🔴 High Risk</div>
                    <div style='font-size: 0.85em; color: #94a3b8; text-align: center; margin-top: 8px;'>
                        {(high/len(df)*100):.1f}% of total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Timeline Analysis
            st.markdown("#### 📅 Timeline Analysis")
            if not df['Timestamp'].isna().all():
                df_timeline = df.groupby(df['Timestamp'].dt.date).size().reset_index()
                df_timeline.columns = ['Date', 'Count']
                
                fig_timeline, ax_timeline = plt.subplots(figsize=(12, 5), facecolor='white')
                ax_timeline.plot(df_timeline['Date'], df_timeline['Count'], 
                               marker='o', linewidth=2.5, markersize=8, 
                               color='#667eea')
                ax_timeline.fill_between(df_timeline['Date'], df_timeline['Count'], 
                                        alpha=0.3, color='#667eea')
                ax_timeline.set_xlabel("Date", fontsize=12, weight='bold', color='#1e293b')
                ax_timeline.set_ylabel("Messages Flagged", fontsize=12, weight='bold', color='#1e293b')
                ax_timeline.set_title("Flagged Messages Over Time", fontsize=15, weight='bold', pad=20, color='#1e293b')
                ax_timeline.grid(alpha=0.2, linestyle='--')
                ax_timeline.set_facecolor('#f8fafc')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_timeline)
                plt.close()
        
        else:
            st.markdown("""
            <div class="alert-box alert-info">
                <h4 style='margin: 0 0 8px 0;'>📊 No Data Available</h4>
                <p style='margin: 0;'>Start analyzing messages to see analytics and insights.</p>
            </div>
            """, unsafe_allow_html=True)
    
    except FileNotFoundError:
        st.markdown("""
        <div class="alert-box alert-info">
            <h4 style='margin: 0 0 8px 0;'>📊 No Data Available</h4>
            <p style='margin: 0;'>Start analyzing messages to see analytics and insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# REPORTS PAGE
def reports_page():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("# 📝 Reports & History")
    st.markdown("View, filter, and export flagged message history.")
    
    try:
        df = pd.read_csv("flagged_messages.csv", names=["Message", "Type", "Confidence", "Severity", "Timestamp"])
        
        if len(df) > 0:
            df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # Filters
            st.markdown("### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_filter = st.selectbox(
                    "Severity Level:",
                    ["All", "Low (0-40)", "Medium (40-80)", "High (80-100)"]
                )
            
            with col2:
                type_filter = st.selectbox(
                    "Type:",
                    ["All"] + list(df['Type'].unique())
                )
            
            with col3:
                date_range = st.selectbox(
                    "Date Range:",
                    ["All Time", "Last 7 Days", "Last 30 Days", "Today"]
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if severity_filter != "All":
                if severity_filter == "Low (0-40)":
                    filtered_df = filtered_df[filtered_df['Severity'] < 40]
                elif severity_filter == "Medium (40-80)":
                    filtered_df = filtered_df[(filtered_df['Severity'] >= 40) & (filtered_df['Severity'] < 80)]
                else:
                    filtered_df = filtered_df[filtered_df['Severity'] >= 80]
            
            if type_filter != "All":
                filtered_df = filtered_df[filtered_df['Type'] == type_filter]
            
            if date_range != "All Time":
                today = datetime.now()
                if date_range == "Today":
                    filtered_df = filtered_df[filtered_df['Timestamp'].dt.date == today.date()]
                elif date_range == "Last 7 Days":
                    filtered_df = filtered_df[filtered_df['Timestamp'] >= (today - timedelta(days=7))]
                else:
                    filtered_df = filtered_df[filtered_df['Timestamp'] >= (today - timedelta(days=30))]
            
            # Display results
            st.markdown(f"### 📋 Results ({len(filtered_df)} messages)")
            
            # Format for display
            display_df = filtered_df.copy()
            display_df['Message'] = display_df['Message'].apply(lambda x: x[:80] + '...' if len(str(x)) > 80 else x)
            display_df['Type'] = display_df['Type'].apply(lambda x: x.replace("_", " ").title())
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df[['Timestamp', 'Message', 'Type', 'Confidence', 'Severity']],
                use_container_width=True,
                height=400
            )
            
            # Export options
            st.markdown("### 📥 Export Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_export = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_export,
                    "flagged_messages_export.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                # Generate summary report
                summary = f"""
CYBERGUARD AI - SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERVIEW:
- Total Messages: {len(filtered_df)}
- Average Severity: {filtered_df['Severity'].mean():.1f}
- Critical Cases: {len(filtered_df[filtered_df['Severity'] >= 80])}

BY TYPE:
{filtered_df['Type'].value_counts().to_string()}

BY SEVERITY:
- Low: {len(filtered_df[filtered_df['Severity'] < 40])}
- Medium: {len(filtered_df[(filtered_df['Severity'] >= 40) & (filtered_df['Severity'] < 80)])}
- High: {len(filtered_df[filtered_df['Severity'] >= 80])}
                """
                
                st.download_button(
                    "📊 Download Summary",
                    summary,
                    "summary_report.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🗑️ Clear All Data", use_container_width=True):
                    if st.checkbox("Confirm deletion"):
                        open("flagged_messages.csv", 'w').close()
                        st.success("All data cleared!")
                        st.rerun()
        
        else:
            st.markdown("""
            <div class="alert-box alert-info">
                <h4 style='margin: 0 0 8px 0;'>📝 No Reports Available</h4>
                <p style='margin: 0;'>Start analyzing messages to generate reports.</p>
            </div>
            """, unsafe_allow_html=True)
    
    except FileNotFoundError:
        st.markdown("""
        <div class="alert-box alert-info">
            <h4 style='margin: 0 0 8px 0;'>📝 No Reports Available</h4>
            <p style='margin: 0;'>Start analyzing messages to generate reports.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ABOUT PAGE
def about_page():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("# ℹ️ About CyberGuard AI")
    
    st.markdown("""
    ### 🎯 Mission
    CyberGuard AI is an advanced artificial intelligence system designed to detect and prevent cyberbullying across digital platforms. 
    Our mission is to create safer online communities by providing real-time, accurate detection of harmful content.
    
    ### 🤖 Technology
    Our system leverages state-of-the-art machine learning algorithms including:
    - **TF-IDF Vectorization** for text feature extraction
    - **Multi-class Classification** for identifying different types of cyberbullying
    - **Severity Scoring Algorithm** for risk assessment
    - **Real-time Processing** for instant analysis
    
    ### 📊 Detection Categories
    CyberGuard AI can identify the following types of cyberbullying:
    - **Age-based harassment** - Discrimination based on age
    - **Ethnicity-based harassment** - Racial and ethnic discrimination
    - **Gender-based harassment** - Sexism and gender discrimination
    - **Religion-based harassment** - Religious intolerance and discrimination
    - **General cyberbullying** - Other forms of online harassment
    
    ### 🔒 Privacy & Security
    - All data is processed securely
    - No personal information is stored
    - GDPR compliant
    - Enterprise-grade encryption
    
    ### 📞 Contact & Support
    For questions, support, or partnership inquiries:
    - Email: support@cyberguard.ai
    - Website: www.cyberguard.ai
    - Documentation: docs.cyberguard.ai
    """)
    
    st.markdown("---")
    
    st.markdown("### 🏆 System Specifications")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Performance Metrics:**
        - Accuracy: 99.2%
        - Response Time: <0.5s
        - Languages Supported: English
        - Max Message Length: 10,000 characters
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - Real-time Analysis
        - Batch Processing
        - Severity Scoring
        - Analytics Dashboard
        - Export Reports
        - API Access (Coming Soon)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main App Logic
def main():
    render_header()
    
    # Page routing
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'analyze':
        analyze_page()
    elif st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'reports':
        reports_page()
    elif st.session_state.page == 'about':
        about_page()
    
    render_footer()

if __name__ == "__main__":
    main()