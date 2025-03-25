import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_firebase():
    """Initialize Firebase connection"""
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS'))
            firebase_admin.initialize_app(cred)
            return True
        except Exception as e:
            st.error(f"Firebase initialization failed: {e}")
            return False

def get_firestore_db():
    """Get Firestore database instance"""
    if initialize_firebase():
        return firestore.client()
    return None