# Required imports for application functionality
import os
import openai
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
import numpy as np
import faiss
from openai.embeddings_utils import get_embedding
from bs4 import BeautifulSoup

# Configure Streamlit page settings - MUST BE FIRST!
st.set_page_config(page_title="ShopRisk", page_icon="📊", layout="wide")

# Initialize session state variables
if 'accepted_terms' not in st.session_state:
    st.session_state.accepted_terms = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'index_ready' not in st.session_state:
    st.session_state.index_ready = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

# Set up sidebar with API key input and navigation
with st.sidebar:
    # Load and display the ShopRisk logo
    st.image('images/shoprisk.jpg', use_column_width=True)
    
    # Navigation menu
    options = option_menu(
        menu_title="Navigation",
        options=["Home", "Risk Analysis", "Data Insights", "Settings"],
        icons=["house", "graph-up", "lightbulb", "gear"],
        menu_icon="cast",
        default_index=0,
    )
    
    # API Key input
    st.markdown('<p style="color: white;">Enter OpenAI API token:</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([5,1], gap="small")
    with col1:
        openai.api_key = st.text_input('', type='password', label_visibility="collapsed")
    with col2:
        check_api = st.button('▶', key='api_button')
    
    if check_api:
        if not openai.api_key:
            st.warning('Please enter your OpenAI API token!')

# Display warning page for first-time users
if not st.session_state.accepted_terms:
    st.markdown("""
        <style>
        .warning-header {
            color: white;
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .warning-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #ff4b4b;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='warning-header'>⚠️ Important Warnings and Guidelines</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='warning-section'>", unsafe_allow_html=True)
    st.markdown("### 1. API Key Security")
    st.markdown("""
    - Keep your OpenAI API key secure and private
    - Never share your API key with others
    - Ensure proper API key format (starts with 'sk-')
    - Monitor API usage and costs
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='warning-section'>", unsafe_allow_html=True)
    st.markdown("### 2. Data Usage Guidelines")
    st.markdown("""
    - Only use accurate and verified delivery data
    - Ensure data is properly formatted
    - Keep sensitive business information confidential
    - Regular data updates recommended
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    agree = st.checkbox("I have read and agree to the above warnings and guidelines")
    if st.button("Continue to ShopRisk", disabled=not agree):
        st.session_state.accepted_terms = True
        st.rerun()
    
    st.stop()

# Options : Home
if options == "Home":
    st.markdown("<h1 style='text-align: center; margin-bottom: 15px; color: white;'>Welcome to ShopRisk!</h1>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; padding: 10px; margin-bottom: 20px; font-size: 18px; color: white;'>ShopRisk is your intelligent companion for e-commerce risk assessment and financial loss prediction. Our AI-powered system analyzes delivery data to help you make informed decisions about your Lazada and Shopee operations in the Philippines.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #FBCD5D; margin-bottom: 10px;'>Key Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; font-size: 16px; color: black; min-height: 200px;'>
        <ul style='list-style-type: none; padding-left: 0; margin: 0;'>
        <li style='margin-bottom: 8px;'>• Financial Loss Prediction</li>
        <li style='margin-bottom: 8px;'>• Risk Assessment Analysis</li>
        <li style='margin-bottom: 8px;'>• Courier Performance Tracking</li>
        <li style='margin-bottom: 8px;'>• Weather Impact Analysis</li>
        <li style='margin-bottom: 8px;'>• Regional Risk Mapping</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)