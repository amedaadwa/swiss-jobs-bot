import os
import base64
import pickle
from email.message import EmailMessage
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from google.cloud import firestore
import json
from datetime import datetime, timedelta
from openai import OpenAI
import io
import extra_streamlit_components as stx  # NEW: For cookie management

# ================================
# SETUP & CONFIGURATION
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
CSV_FILE = "Assistenzarzt_Jobs_CH__Combined_Final.csv"
SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/userinfo.email"]
DB_COLLECTION = "job_applications_v3"

# NEW: Define redirect URI (update for your hosted app URL; use environment variable if possible)
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")  # e.g., "https://your-app.streamlit.app" for hosted

# ================================
# FIREBASE AUTHENTICATION (Unchanged)
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

# ================================
# GMAIL AUTHENTICATION (Major Update: Web-based flow for hosted env)
# ================================
def get_auth_url():
    try:
        client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
        if not client_secret_json:
            st.error("Google client secret JSON not found.")
            return None
        client_config = json.loads(client_secret_json)
        flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
        auth_url, _ = flow.authorization_url(access_type='offline', include_granted_scopes='true')
        return auth_url
    except Exception as e:
        st.error(f"Failed to generate auth URL: {e}")
        return None

def load_creds(db, user_email):
    try:
        doc = db.collection(DB_COLLECTION).document(user_email).get()
        if doc.exists:
            data = doc.to_dict()
            token_b64 = data.get('gmail_token')
            if token_b64:
                creds = pickle.loads(base64.b64decode(token_b64))
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    new_token_b64 = base64.b64encode(pickle.dumps(creds)).decode('utf-8')
                    db.collection(DB_COLLECTION).document(user_email).set({'gmail_token': new_token_b64}, merge=True)
                return creds
    except Exception as e:
        st.error(f"Failed to load/refresh credentials: {e}")
    return None

# ================================
# API & HELPER FUNCTIONS (Unchanged)
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
    if not cv_content:
        st.error("CV content is missing. Cannot generate email.")
        return {'subject': '', 'body': ''}
    
    job_details_prompt = f"- Position: {job_title}\n- Hospital: {hospital_name}\n- Canton: {canton}\n- Job Description:\n---\n{job_description}\n---" if job_title and hospital_name else f"- Job Details:\n---\n{job_description}\n---"
    prompt = f"""
Act as a professional medical career advisor in Switzerland. Create a compelling application email in German.
**Applicant's Profile (from CV):**
---
{cv_content}
---
**Job Details:**
{job_details_prompt}
**Instructions:**
1. **Generate Subject:** Create a concise, professional German subject line.
2. **Generate Body:** Write a polite, personalized email connecting the applicant's CV to the job.
3. **Output Format:** Output MUST be 'Subject: [Your Subject]|||Body: [Your Body]'.
"""
    response = call_openai_api(prompt, "You are a professional medical job applicant assistant, writing in German.")
    if response and '|||' in response:
        subject = response.split('|||')[0].replace('Subject:', '').strip()
        body = response.split('|||')[1].replace('Body:', '').strip()
        return {'subject': subject, 'body': body}
    
    return {'subject': "Bewerbung für die ausgeschriebene Position", 'body': response or "Could not generate email body."}

def send_email_logic(service, user_email, to_email, subject, body, attachments):
    """
    Sends an email using the Gmail API from the authenticated user's email address.
    """
    try:
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to_email
        message["Subject"] = subject
        message["From"] = user_email  # Set From to the authenticated user's email

        for file_wrapper in attachments:
            file_content = base64.b64decode(file_wrapper['content_b64'])
            message.add_attachment(file_content, maintype="application", subtype="octet-stream", filename=file_wrapper['name'])
        
        encoded_message = base64.b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        service.users().messages().send(userId="me", body=create_message).execute()
        st.success(f"✅ Application successfully sent to {to_email}!")
        return True
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}. If credentials expired, please re-authenticate.")
        return False

def extract_text_from_pdf(file_bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def translate_cv_text(text):
    prompt = f"Please translate the following CV text from English to professional, high-quality German suitable for a medical job application in Switzerland.\n\n**Text to Translate:**\n---\n{text}\n---"
    return call_openai_api(prompt, "You are an expert translator specializing in medical and professional documents.")

# ================================
# DATABASE FUNCTIONS (Unchanged)
# ================================
def get_user_data(db, user_email):
    if not db or not user_email: return {}
    doc = db.collection(DB_COLLECTION).document(user_email).get()
    return doc.to_dict() if doc.exists else {}

def update_user_data(db, user_email, data_to_update):
    if not db or not user_email: return
    db.collection(DB_COLLECTION).document(user_email).set(data_to_update, merge=True)

def save_sent_email(db, user_email, email_data):
    if not db or not user_email: return
    db.collection(DB_COLLECTION).document(user_email).collection("sent_emails").add(email_data)

# ================================
# UI PAGE FUNCTIONS (Unchanged)
# ================================
def render_dashboard(db, user_email):
    st.header("📊 Dashboard: Sent Applications")
    emails_ref = db.collection(DB_COLLECTION).document(user_email).collection("sent_emails").order_by("sent_at", direction=firestore.Query.DESCENDING).stream()
    
    emails = list(emails_ref)
    st.metric("Total Emails Sent", len(emails))
    if not emails:
        st.info("You haven't sent any emails yet. Head over to the 'Job Finder' to get started!")
        return
    for email in emails:
        data = email.to_dict()
        sent_time = data.get('sent_at', datetime.now()).strftime("%d %b %Y, %H:%M")
        with st.expander(f"To: {data['recipient']} | Subject: {data['subject']} | Sent: {sent_time}"):
            st.write(f"**To:** {data['recipient']}")
            st.write(f"**Subject:** {data['subject']}")
            st.write(f"**Sent At:** {sent_time}")
            st.text_area("Email Body", value=data['body'], height=300, disabled=True, key=f"body_{email.id}")

def render_job_finder(db, user_email):
    st.header("🔍 Job Finder")
    try:
        jobs_df = pd.read_csv(CSV_FILE)
        user_data = get_user_data(db, user_email)
        applied_jobs = set(user_data.get("applied_jobs", []))
    except FileNotFoundError:
        st.error(f"Error: '{CSV_FILE}' not found."); st.stop()
    
    next_job = next(( (idx, row) for idx, row in jobs_df.iterrows() if str(idx) not in applied_jobs and isinstance(row.get("Application Contact Email"), str) and "@" in row.get("Application Contact Email", "") ), None)
        
    if next_job is None:
        st.info("🎉 All jobs from the CSV have been processed!"); return
    job_id, row_data = next_job
    
    if st.session_state.current_job_id != job_id:
        st.session_state.current_job_id = job_id
        st.session_state.generated_email_content = None
        st.session_state.current_job_details = {
            "job_id": job_id, "job_title": row_data.get("job_title", "N/A"),
            "hospital_name": row_data.get("hospital_name", ""), "canton": row_data.get("canton", ""),
            "contact_email": str(row_data.get("Application Contact Email", "")).strip().split(",")[0],
            "job_description": row_data.get("Job Description (short)", "")
        }
    details = st.session_state.current_job_details
    st.subheader(f"Next Up: {details['job_title']}")
    st.write(f"**Hospital:** {details['hospital_name']} ({details['canton']})")
    col1, col2 = st.columns([3, 1])
    if col1.button(f"🤖 Prepare Application for Job #{details['job_id']}"):
        with st.spinner("Generating personalized email..."):
            st.session_state.generated_email_content = generate_personalized_email(
                st.session_state.cv_text, details['job_title'], details['hospital_name'],
                details['canton'], details['job_description']
            )
    if col2.button("Skip Job ⏭️"):
        update_user_data(db, user_email, {"applied_jobs": firestore.ArrayUnion([str(details['job_id'])])})
        db.collection(DB_COLLECTION).document(user_email).update({'stats.skipped_count': firestore.Increment(1)})
        st.warning(f"Skipped job #{details['job_id']}."); st.session_state.current_job_id = None; st.rerun()
    if st.session_state.get('generated_email_content'):
        render_application_form(db, user_email)

def render_application_form(db, user_email, is_manual=False):
    st.markdown("---")
    st.subheader("✉️ Review, Edit, and Send Application")
    details, email_content, form_key = ( (st.session_state.manual_job_details, st.session_state.manual_email_content, "manual_form") if is_manual
                                         else (st.session_state.current_job_details, st.session_state.generated_email_content, f"form_{st.session_state.current_job_details['job_id']}") )
    with st.form(key=form_key):
        contact_email = st.text_input("To", value=details.get('contact_email', ''))
        subject = st.text_input("Subject", value=email_content.get('subject', ''))
        body = st.text_area("Email Body", value=email_content.get('body', ''), height=350)
        
        st.write("**Attachments:**")
        for att in st.session_state.attachments:
            st.info(f"📄 {att['name']}")
        
        send_button = st.form_submit_button("🚀 Send Application")
    if send_button:
        if not st.session_state.attachments:
            st.warning("You must have at least one attachment (like a CV). Go to Settings to upload."); return
        if send_email_logic(st.session_state.gmail_service, st.session_state.user_email, contact_email, subject, body, st.session_state.attachments):
            save_sent_email(db, user_email, {
                "recipient": contact_email, "subject": subject, "body": body,
                "sent_at": firestore.SERVER_TIMESTAMP, "job_title": details.get('job_title', 'Manual'),
            })
            if not is_manual:
                update_user_data(db, user_email, {"applied_jobs": firestore.ArrayUnion([str(details['job_id'])])})
                db.collection(DB_COLLECTION).document(user_email).update({'stats.sent_count': firestore.Increment(1)})
                st.session_state.current_job_id = None
            else:
                st.session_state.manual_email_content = None
            st.balloons(); st.rerun()

def render_settings_page(db, user_email):
    st.header("⚙️ Settings & Profile")
    st.subheader("Manage Your Attachments")
    if st.session_state.attachments:
        st.write("Current Attachments:")
        for att in st.session_state.attachments:
            st.success(f"📄 {att['name']}")
    else:
        st.info("No attachments found. Please upload your CV.")
    st.markdown("---")
    st.subheader("Upload New CV and Attachments")
    st.warning("Uploading new files will **replace all** existing ones.")
    
    uploaded_files = st.file_uploader(
        "Upload your CV (must be first) and other files.",
        accept_multiple_files=True,
        key="settings_uploader"
    )
    if uploaded_files:
        if st.button("Save New Files"):
            process_and_save_files(db, user_email, uploaded_files)

def process_and_save_files(db, user_email, files):
    with st.spinner("Processing and saving files..."):
        attachments_data = []
        cv_text = ""
        cv_file = files[0]
        cv_bytes = cv_file.getvalue()
        attachments_data.append({
            "name": cv_file.name,
            "content_b64": base64.b64encode(cv_bytes).decode('utf-8')
        })
        
        raw_cv_text = extract_text_from_pdf(cv_bytes)
        if raw_cv_text:
            if any(char in 'àéâç' for char in raw_cv_text[:500]):
                cv_text = raw_cv_text
            else:
                cv_text = translate_cv_text(raw_cv_text)
        
        for file in files[1:]:
            attachments_data.append({
                "name": file.name,
                "content_b64": base64.b64encode(file.getvalue()).decode('utf-8')
            })
        
        update_user_data(db, user_email, {
            "attachments": attachments_data,
            "cv_text": cv_text
        })
        st.success("Your files and CV have been saved!")
        st.session_state.attachments = attachments_data
        st.session_state.cv_text = cv_text
        st.rerun()

# ================================
# MAIN APP LAYOUT & LOGIC (Updated for web-based auth)
# ================================
st.set_page_config(layout="wide")
st.title("🇨🇭 Swiss Assistenzarzt Job Application Bot")

# Initialize session state
default_states = {
    'user_email': None, 'gmail_service': None, 'cv_text': None,
    'attachments': [], 'current_job_id': None, 'db': None,
    'generated_email_content': None, 'manual_email_content': None,
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Connect to Firestore
if not st.session_state.db:
    st.session_state.db = get_firestore_db()
if not st.session_state.db:
    st.error("Could not connect to the database. The app cannot continue.")
    st.stop()

# NEW: Cookie manager for persistence
cookie_manager = stx.CookieManager()

# Primary App Flow
if not st.session_state.user_email:
    # Try loading from cookie
    user_email_cookie = cookie_manager.get('user_email')
    if user_email_cookie:
        creds = load_creds(st.session_state.db, user_email_cookie)
        if creds:
            st.session_state.gmail_service = build("gmail", "v1", credentials=creds)
            st.session_state.user_email = user_email_cookie
            st.rerun()

if not st.session_state.user_email:
    st.header("Step 1: Authorize Your Gmail Account")
    st.info("This app needs permission to send emails on your behalf and view your email address to create your profile.")
    
    # Generate and show auth link
    auth_url = get_auth_url()
    if auth_url:
        st.link_button("Login with Google", auth_url)
    
    # Check for authorization code in query params (callback)
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        try:
            client_config = json.loads(os.getenv("GOOGLE_CLIENT_SECRET_JSON"))
            flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
            flow.fetch_token(code=code)
            creds = flow.credentials
            service = build("gmail", "v1", credentials=creds)
            profile = service.users().getProfile(userId="me").execute()
            user_email = profile['emailAddress']
            token_b64 = base64.b64encode(pickle.dumps(creds)).decode('utf-8')
            st.session_state.db.collection(DB_COLLECTION).document(user_email).set({'gmail_token': token_b64}, merge=True)
            cookie_manager.set('user_email', user_email, expires_at=datetime.now() + timedelta(days=30))
            st.experimental_set_query_params()  # Clear query params
            st.session_state.gmail_service = service
            st.session_state.user_email = user_email
            st.success(f"Logged in as {user_email}")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    user_data = get_user_data(st.session_state.db, st.session_state.user_email)
    st.session_state.cv_text = user_data.get("cv_text")
    st.session_state.attachments = user_data.get("attachments", [])
    
    if not st.session_state.cv_text or not st.session_state.attachments:
        st.header("Step 2: Upload Your Documents")
        st.info("Please upload your CV. The first file will be treated as your CV and its text will be extracted for generating emails.")
        uploaded_files = st.file_uploader(
            "Upload your CV (must be first) and any other attachments.",
            accept_multiple_files=True,
            key="initial_uploader"
        )
        if uploaded_files:
            if st.button("Confirm and Save Files"):
                process_and_save_files(st.session_state.db, st.session_state.user_email, uploaded_files)
    else:
        with st.sidebar:
            st.header(f"Welcome!")
            st.write(st.session_state.user_email)
            st.markdown("---")
            app_page = st.radio("Navigation", ["Job Finder", "Dashboard", "Settings"])
            st.markdown("---")
            stats = user_data.get("stats", {})
            st.header("📊 Your Stats")
            st.metric("Emails Sent", stats.get("sent_count", 0))
            st.metric("Jobs Skipped", stats.get("skipped_count", 0))
        
        if app_page == "Job Finder":
            render_job_finder(st.session_state.db, st.session_state.user_email)
        elif app_page == "Dashboard":
            render_dashboard(st.session_state.db, st.session_state.user_email)
        elif app_page == "Settings":
            render_settings_page(st.session_state.db, st.session_state.user_email)