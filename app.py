import os
import base64
import pickle
from email.message import EmailMessage
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from google.cloud import firestore
import json
from datetime import datetime
from openai import OpenAI

# ================================
# SETUP & CONFIGURATION
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

CSV_FILE = "Assistenzarzt_Jobs_CH__Combined_Final.csv"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]
DB_COLLECTION = "job_applications_v2"

# ================================
# FIREBASE & GMAIL AUTHENTICATION
# ================================
@st.cache_resource
def get_firestore_db():
    try:
        firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not firebase_creds_json:
            st.error("Firebase service account JSON not found.")
            return None
        creds_dict = json.loads(firebase_creds_json)
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        return firestore.Client(credentials=creds)
    except Exception as e:
        st.error(f"Failed to connect to Firebase: {e}")
        return None


def _pickle_creds(creds):
    return base64.b64encode(pickle.dumps(creds)).decode("utf-8")


def _unpickle_creds(token_b64):
    return pickle.loads(base64.b64decode(token_b64.encode("utf-8")))


def gmail_authenticate():
    """Authenticate user with Gmail and return service, email, creds."""
    db = get_firestore_db()
    if not db:
        st.error("No DB connection.")
        return None, None, None

    client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
    if not client_secret_json:
        st.error("Google client secret JSON not found in env.")
        return None, None, None

    client_config = json.loads(client_secret_json)

    try:
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        st.error(f"Interactive Google auth failed: {e}")
        return None, None, None

    try:
        temp_service = build("gmail", "v1", credentials=creds)
        profile = temp_service.users().getProfile(userId="me").execute()
        user_email = profile.get("emailAddress")
        if not user_email:
            st.error("Could not determine authenticated user's email.")
            return None, None, None
    except Exception as e:
        st.error(f"Failed to get Gmail profile: {e}")
        return None, None, None

    try:
        token_b64 = _pickle_creds(creds)
        db.collection(DB_COLLECTION).document(user_email).set(
            {"gmail_token": token_b64, "auth_provider": "oauth_user"},
            merge=True,
        )
    except Exception as e:
        st.warning(f"Token save warning (non-fatal): {e}")

    try:
        service = build("gmail", "v1", credentials=creds)
        return service, user_email, creds
    except Exception as e:
        st.error(f"Failed to build Gmail service: {e}")
        return None, None, None

# ================================
# API & HELPER FUNCTIONS
# ================================
def call_openai_api(prompt, system_message="You are a helpful assistant."):
    try:
        if not OPENAI_API_KEY:
            st.error("OpenAI API key is not configured.")
            return None
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None


def generate_personalized_email(cv_content, job_title=None, hospital_name=None, canton=None, job_description=None):
    if job_title and hospital_name:
        job_details_prompt = f"""
- Position: {job_title}
- Hospital: {hospital_name}
- Canton: {canton}
- Job Description:
---
{job_description}
---
"""
    else:
        job_details_prompt = f"""
- Job Details (extract relevant info from this block):
---
{job_description}
---
"""

    prompt = f"""
Act as a professional medical career advisor in Switzerland.
Your task is to create a compelling application email in German.

**Applicant's Profile (from CV):**
---
{cv_content}
---
**Job Details:**
{job_details_prompt}

**Instructions:**
1. Identify job title and hospital name if needed.
2. Generate a German subject line.
3. Generate a professional email body.
4. Return in the format: Subject|||Body
"""

    response = call_openai_api(prompt, "You are a professional medical job applicant assistant, writing in German.")
    if response and '|||' in response:
        parts = response.split('|||', 1)
        subject = parts[0].replace('Betreff:', '').replace('Subject:', '').strip()
        body = parts[1].strip()
        return {'subject': subject, 'body': body}

    return {'subject': f"Bewerbung für die ausgeschriebene Position", 'body': response or "Could not generate email body."}


def translate_cv_text(text):
    prompt = f"Translate the following CV text to professional German for a Swiss medical job:\n\n{text}"
    system_message = "You are an expert medical translator."
    return call_openai_api(prompt, system_message)


def send_email_logic(service, to_email, subject, body, attachments, user_email, sender_display_name=None, reply_to=None):
    try:
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to_email
        message["Subject"] = subject
        message["From"] = f'{sender_display_name} <{user_email}>' if sender_display_name else user_email
        if reply_to:
            message["Reply-To"] = reply_to

        for file_wrapper in attachments:
            file_content = base64.b64decode(file_wrapper["content_b64"])
            message.add_attachment(
                file_content,
                maintype="application",
                subtype="octet-stream",
                filename=file_wrapper["name"],
            )

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}

        service.users().messages().send(userId="me", body=create_message).execute()
        st.success(f"✅ Application successfully sent to {to_email}!")
        return True
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}")
        return False


def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# ================================
# DATABASE FUNCTIONS
# ================================
def get_user_data(db, user_email):
    if not db or not user_email:
        return {}
    doc = db.collection(DB_COLLECTION).document(user_email).get()
    return doc.to_dict() if doc.exists else {}


def update_user_data(db, user_email, data_to_update):
    if not db or not user_email:
        return
    db.collection(DB_COLLECTION).document(user_email).set(data_to_update, merge=True)


def save_sent_email(db, user_email, email_data):
    if not db or not user_email:
        return
    db.collection(DB_COLLECTION).document(user_email).collection("sent_emails").add(email_data)

# ================================
# UI PAGE FUNCTIONS
# ================================
def render_dashboard(db):
    st.header("📊 Dashboard: Sent Applications")
    emails_ref = db.collection(DB_COLLECTION).document(st.session_state.user_email).collection("sent_emails") \
        .order_by("sent_at", direction=firestore.Query.DESCENDING).stream()

    emails = list(emails_ref)
    if not emails:
        st.info("You haven't sent any emails yet. Head over to the 'Job Finder' to get started!")
        return

    for email in emails:
        data = email.to_dict()
        sent_time = data['sent_at'].strftime("%d %b %Y, %H:%M") if isinstance(data['sent_at'], datetime) else "Unknown"
        with st.expander(f"To: {data['recipient']} | Subject: {data['subject']} | Sent: {sent_time}"):
            st.write(f"**To:** {data['recipient']}")
            st.write(f"**Subject:** {data['subject']}")
            st.write(f"**Sent At:** {sent_time}")
            st.markdown("---")
            st.text_area("Email Body", value=data['body'], height=300, disabled=True, key=f"body_{email.id}")

# ================================
# MAIN APP
# ================================
st.set_page_config(layout="wide")
st.title("🇨🇭 Swiss Assistenzarzt Job Application Bot")

default_states = {
    'user_email': None, 'step': 'auth', 'gmail_service': None, 'cv_content': None,
    'attachments': [], 'current_job_id': None,
    'generated_email_content': None, 'manual_email_content': None,
    'db': None
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

if not st.session_state.db:
    st.session_state.db = get_firestore_db()
if not st.session_state.db:
    st.stop()

if st.session_state.step == "auth":
    st.header("Step 1: Authorize Your Gmail Account")
    if st.button("Authorize Gmail"):
        with st.spinner("Authenticating..."):
            service, email, creds = gmail_authenticate()
        if service:
            st.session_state.gmail_service = service
            st.session_state.user_email = email
            st.session_state.step = "upload_cv"
            st.rerun()
