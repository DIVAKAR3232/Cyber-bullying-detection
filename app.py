import streamlit as st
import joblib
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import time

# Load model and vectorizer
model = joblib.load("cyberbullying_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Abusive words for highlighting
abusive_words = ["idiot", "stupid", "hate", "dumb", "loser", "kill", "ugly", "die", "pathetic", "worthless"]

# Severity scoring system
def calculate_severity(prediction, confidence, text):
    """Calculate severity score based on prediction type, confidence, and content"""
    base_scores = {
        "not_cyberbullying": 0,
        "age": 60,
        "ethnicity": 85,
        "gender": 75,
        "religion": 80,
        "other_cyberbullying": 70
    }
    
    # Get base score
    base = base_scores.get(prediction.lower(), 50)
    
    # Adjust based on confidence
    confidence_factor = confidence / 100
    
    # Count abusive words
    abusive_count = sum(1 for word in abusive_words if re.search(rf"\b{word}\b", text, re.IGNORECASE))
    abusive_factor = min(abusive_count * 5, 20)
    
    # Calculate final severity (0-100)
    severity = min(100, base * confidence_factor + abusive_factor)
    
    return round(severity, 1)

def get_severity_level(score):
    """Get severity level and color based on score"""
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
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Page config with custom theme
st.set_page_config(
    page_title="🛡️ CyberGuard - AI Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional white background with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Clean white */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Animated background pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
        animation: bgFloat 20s ease-in-out infinite;
        z-index: 0;
        pointer-events: none;
    }
    
    @keyframes bgFloat {
        0%, 100% { transform: scale(1) translateY(0); }
        50% { transform: scale(1.1) translateY(-20px); }
    }
    
    /* Container styling with glass effect */
    .main-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.08),
            0 0 0 1px rgba(0, 0, 0, 0.02),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        margin: 20px 0;
        animation: containerFadeIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes containerFadeIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5em;
        font-weight: 800;
        margin-bottom: 10px;
        animation: headerFloat 3s ease-in-out infinite;
    }
    
    @keyframes headerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.3em;
        font-weight: 400;
        margin-bottom: 30px;
        animation: fadeInUp 1s ease-in;
    }
    
    /* Button styling with animation */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 18px 48px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 
            0 4px 20px rgba(102, 126, 234, 0.4),
            0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 8px 30px rgba(102, 126, 234, 0.5),
            0 3px 8px rgba(0, 0, 0, 0.12);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Metric cards with enhanced design */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.06),
            0 0 0 1px rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: cardSlideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
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
    
    .metric-card:hover::after {
        transform: scaleX(1);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.12),
            0 0 0 1px rgba(0, 0, 0, 0.03);
    }
    
    @keyframes cardSlideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 20px;
    }
    
    /* Feature cards in sidebar */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
    }
    
    .feature-card:hover {
        border-left-color: #667eea;
        transform: translateX(8px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        font-size: 16px;
        padding: 20px;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* Alert boxes with animation */
    .alert-box {
        padding: 24px;
        border-radius: 16px;
        margin: 20px 0;
        animation: alertSlideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes alertSlideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border-left: 6px solid #10b981;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-left: 6px solid #ef4444;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-left: 6px solid #f59e0b;
    }
    
    /* Severity gauge styling */
    .severity-gauge {
        position: relative;
        height: 20px;
        background: #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        margin: 16px 0;
    }
    
    .severity-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .severity-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: gaugeShine 2s infinite;
    }
    
    @keyframes gaugeShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Loading animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Stats badge */
    .stats-badge {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        animation: badgeBounce 2s ease-in-out infinite;
    }
    
    @keyframes badgeBounce {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animations
lottie_shield = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_kxsd2ytq.json")
lottie_analyzing = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_p8bfn5to.json")

# Header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if lottie_shield:
        st_lottie(lottie_shield, height=180, key="shield")

st.markdown('<h1 class="main-header">🛡️ CyberGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Cyberbullying Detection & Analysis System</p>', unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("### 🎯 System Features")
    
    features = [
        ("🔍", "Multi-Class Detection", "Identifies 5+ types of harmful content"),
        ("⚡", "Real-Time Analysis", "Instant AI-powered classification"),
        ("📊", "Severity Scoring", "0-100 risk assessment scale"),
        ("💡", "Smart Highlighting", "Marks problematic language"),
        ("💾", "Auto-Logging", "Secure message archiving"),
        ("📈", "Visual Analytics", "Interactive insights dashboard")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <h4 style='margin: 0; color: #1e293b;'>{icon} {title}</h4>
            <p style='font-size: 0.85em; color: #64748b; margin: 8px 0 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    try:
        df = pd.read_csv("flagged_messages.csv", names=["Message", "Type", "Confidence", "Severity", "Timestamp"])
        total = len(df)
        critical = len(df[df["Severity"].astype(float) >= 80]) if len(df) > 0 else 0
        
        st.markdown(f'<div class="stats-badge">Total Flagged: {total}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stats-badge" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">Critical: {critical}</div>', unsafe_allow_html=True)
        
        if len(df) > 0:
            most_common = df["Type"].mode()[0].replace("_", " ").title()
            st.info(f"🔥 Most Common: **{most_common}**")
    except:
        st.markdown('<div class="stats-badge">Total Flagged: 0</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# User Input section
st.markdown("### 📝 Message Analysis")
st.write("Enter any text message, comment, or social media post to analyze for cyberbullying content.")

user_input = st.text_area(
    "",
    placeholder="Type or paste your message here for instant AI analysis...",
    height=150,
    label_visibility="collapsed"
)

def highlight_abusive(text):
    for word in abusive_words:
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        text = pattern.sub(f"**:red[{word.upper()}]**", text)
    return text

# Create columns for button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Analyze Message", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Display results
if predict_button:
    if user_input.strip() == "":
        st.markdown("""
        <div class="alert-box alert-warning">
            <h3 style='margin: 0 0 8px 0;'>⚠️ Input Required</h3>
            <p style='margin: 0;'>Please enter a message to analyze.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show analyzing animation
        with st.spinner(''):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            vect_input = vectorizer.transform([user_input])
            prediction = model.predict(vect_input)[0]
            prediction_proba = np.max(model.predict_proba(vect_input)) * 100
            
            # Calculate severity score
            severity_score = calculate_severity(prediction, prediction_proba, user_input)
            severity_level, severity_color, severity_icon = get_severity_level(severity_score)
            
            progress_bar.empty()
        
        # Results card
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Display highlighted text
        st.markdown("### 🔎 Analysis Results")
        highlighted_text = highlight_abusive(user_input)
        st.markdown(f"**Message Preview:** {highlighted_text}")
        
        st.markdown("---")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size: 0.85em; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>Classification</div>
                <div style='font-size: 1.8em; font-weight: 700; color: #1e293b; margin-bottom: 4px;'>{prediction.replace("_", " ").title()}</div>
                <div style='font-size: 0.75em; color: #94a3b8;'>AI Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_color = "#10b981" if prediction_proba > 70 else "#f59e0b"
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size: 0.85em; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>Confidence</div>
                <div style='font-size: 1.8em; font-weight: 700; color: {confidence_color}; margin-bottom: 4px;'>{prediction_proba:.1f}%</div>
                <div style='font-size: 0.75em; color: #94a3b8;'>Model Certainty</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size: 0.85em; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>Severity Score</div>
                <div style='font-size: 1.8em; font-weight: 700; color: {severity_color}; margin-bottom: 4px;'>{severity_icon} {severity_score}</div>
                <div style='font-size: 0.75em; color: #94a3b8;'>Risk Level: {severity_level}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = "SAFE" if prediction.lower() == "not_cyberbullying" else "FLAGGED"
            status_color = "#10b981" if status == "SAFE" else "#ef4444"
            status_icon = "✅" if status == "SAFE" else "⚠️"
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size: 0.85em; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>Status</div>
                <div style='font-size: 1.8em; font-weight: 700; color: {status_color}; margin-bottom: 4px;'>{status_icon}</div>
                <div style='font-size: 0.75em; color: #94a3b8;'>{status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Severity gauge
        st.markdown("#### Severity Assessment")
        st.markdown(f"""
        <div class="severity-gauge">
            <div class="severity-fill" style="width: {severity_score}%; background: linear-gradient(90deg, {severity_color}, {severity_color}dd);"></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 0.75em; color: #64748b; margin-top: 4px;'>
            <span>0 (Safe)</span>
            <span>50 (Moderate)</span>
            <span>100 (Critical)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Log flagged messages
        if prediction.lower() != "not_cyberbullying":
            with open("flagged_messages.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([user_input, prediction, f"{prediction_proba:.2f}%", severity_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
            st.markdown(f"""
            <div class="alert-box alert-danger">
                <h3 style='margin: 0 0 12px 0;'>⚠️ Message Flagged - {severity_level} Risk</h3>
                <p style='margin: 0; line-height: 1.6;'>
                    This message has been logged for review. Our AI detected potentially harmful content with a 
                    severity score of <strong>{severity_score}/100</strong>. 
                    {f'This is classified as <strong>{severity_level}</strong> risk.' if severity_level != 'SAFE' else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <h3 style='margin: 0 0 12px 0;'>✅ Message Safe</h3>
                <p style='margin: 0; line-height: 1.6;'>
                    No cyberbullying content detected. This message appears to be respectful and appropriate.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analytics section
        try:
            df = pd.read_csv("flagged_messages.csv", names=["Message", "Type", "Confidence", "Severity", "Timestamp"])
            
            if len(df) > 0:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 📊 Cyberbullying Analytics Dashboard")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart for types
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
                    ax_pie.set_title("Distribution by Type", fontsize=15, weight='bold', pad=20, color='#1e293b')
                    for autotext in autotexts:
                        autotext.set_color('white')
                    plt.tight_layout()
                    st.pyplot(fig_pie)
                    plt.close()
                
                with col2:
                    # Bar chart
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
                    ax_bar.set_xlabel("Type", fontsize=12, weight='bold', color='#1e293b')
                    ax_bar.set_title("Flagged Messages Count", fontsize=15, weight='bold', pad=20, color='#1e293b')
                    ax_bar.grid(axis='y', alpha=0.2, linestyle='--', linewidth=1)
                    ax_bar.set_facecolor('#f8fafc')
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontsize=11, weight='bold', color='#1e293b')
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close()
                
                # Severity distribution
                st.markdown("#### Severity Distribution")
                col1, col2, col3 = st.columns(3)
                
                df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
                
                with col1:
                    low = len(df[df['Severity'] < 40])
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #3b82f6;">
                        <div style='font-size: 2em; font-weight: 700; color: #3b82f6;'>{low}</div>
                        <div style='font-size: 0.9em; color: #64748b;'>🔵 Low Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    medium = len(df[(df['Severity'] >= 40) & (df['Severity'] < 80)])
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                        <div style='font-size: 2em; font-weight: 700; color: #f59e0b;'>{medium}</div>
                        <div style='font-size: 0.9em; color: #64748b;'>🟡 Medium Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    high = len(df[df['Severity'] >= 80])
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #ef4444;">
                        <div style='font-size: 2em; font-weight: 700; color: #ef4444;'>{high}</div>
                        <div style='font-size: 0.9em; color: #64748b;'>🔴 High Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown("""
            <div class="alert-box" style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); color: #3730a3; border-left: 6px solid #6366f1;">
                <h4 style='margin: 0 0 8px 0;'>📊 Analytics Dashboard</h4>
                <p style='margin: 0;'>Analytics will appear after messages are flagged and analyzed.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: white; border-radius: 20px; margin: 20px 0;'>
    <h3 style='color: #1e293b; margin-bottom: 12px;'>🛡️ CyberGuard AI</h3>
    <p style='color: #64748b; font-size: 1em; margin-bottom: 8px;'>Powered by Advanced Machine Learning</p>
    <p style='color: #94a3b8; font-size: 0.85em;'>Built with ❤️ for a Safer Digital World</p>
    <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid #e2e8f0;'>
        <span style='color: #94a3b8; font-size: 0.8em;'>© 2024 CyberGuard Systems | Protecting Communities Worldwide</span>
    </div>
</div>
""", unsafe_allow_html=True)