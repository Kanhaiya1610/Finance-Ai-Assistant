import streamlit as st
from firebase_config import get_firestore_db

def test_firebase():
    st.title("🔥 Firebase Connection Test")
    
    db = get_firestore_db()
    if db:
        try:
            # Test write
            doc_ref = db.collection('test').document('test_doc')
            doc_ref.set({
                'message': 'Hello from Finance AI Assistant!'
            })
            
            # Test read
            doc = doc_ref.get()
            if doc.exists:
                st.success("✅ Firebase connection successful!")
                st.json(doc.to_dict())
                
                # Cleanup
                doc_ref.delete()
            
        except Exception as e:
            st.error(f"Test failed: {e}")
    else:
        st.error("Failed to initialize Firebase")

if __name__ == "__main__":
    test_firebase()
