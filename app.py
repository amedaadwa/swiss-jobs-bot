# Redeploy trigger - 2025-10-07


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
import io # NEW: Needed to handle file data in memory

# ================================
# SETUP & CONFIGURATION
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

CSV_FILE = "Assistenzarzt_Jobs_CH__Combined_Final.csv"
SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/userinfo.email"] # MODIFIED: Added scope to get user email
DB_COLLECTION = "job_applications_v3" # MODIFIED: Using a new collection version for clarity

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

def gmail_authenticate():
    """
    Handles Google Authentication.
    Returns (gmail_service, user_email) of the signed-in user.
    Each user must go through OAuth — this ensures emails send from their account.
    """
    db = get_firestore_db()
    creds = None

    try:
        client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
        if not client_secret_json:
            st.error("Google client secret JSON not found.")
            return None, None

        client_config = json.loads(client_secret_json)
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(port=0)

        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        user_email = profile["emailAddress"]

        # Save token per user (optional, can just reauth every time)
        if db:
            token_b64 = base64.b64encode(pickle.dumps(creds)).decode("utf-8")
            db.collection(DB_COLLECTION).document(user_email).set(
                {"gmail_token": token_b64}, merge=True
            )

        return service, user_email

    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None, None

# ================================
# API & HELPER FUNCTIONS (No significant changes here)
# ================================
def call_openai_api(prompt, system_message="You are a helpful assistant."):
    """Generic function to call the OpenAI API."""
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
1.  **Generate Subject:** Create a concise, professional German subject line.
2.  **Generate Body:** Write a polite, personalized email connecting the applicant's CV to the job.
3.  **Output Format:** Output MUST be 'Subject: [Your Subject]|||Body: [Your Body]'.
"""
    response = call_openai_api(prompt, "You are a professional medical job applicant assistant, writing in German.")
    if response and '|||' in response:
        subject = response.split('|||')[0].replace('Subject:', '').strip()
        body = response.split('|||')[1].replace('Body:', '').strip()
        return {'subject': subject, 'body': body}
    
    return {'subject': "Bewerbung für die ausgeschriebene Position", 'body': response or "Could not generate email body."}

def send_email_logic(service, to_email, subject, body, attachments, from_email=None):
    """
    Sends an email from the authenticated user's Gmail account.
    If from_email is provided, it must equal the authenticated user's email or a verified alias.
    """
    try:
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to_email
        message["Subject"] = subject

        # Use the authenticated email as the From header, or omit it
        if from_email:
            message["From"] = from_email  # must match signed-in user or alias

        for file_wrapper in attachments:
            file_content = base64.b64decode(file_wrapper["content_b64"])
            message.add_attachment(
                file_content,
                maintype="application",
                subtype="octet-stream",
                filename=file_wrapper["name"]
            )

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}

        service.users().messages().send(userId="me", body=create_message).execute()
        st.success(f"✅ Application successfully sent to {to_email}!")
        return True
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}")
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
# MODIFIED DATABASE FUNCTIONS (Now user-specific)
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
# UI PAGE FUNCTIONS (MODIFIED for user-specific data)
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

        if send_email_logic(
    st.session_state.gmail_service,
    contact_email,
    subject,
    body,
    st.session_state.attachments,
    from_email=st.session_state.user_email
):

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

# NEW: Settings page to manage CV
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


# NEW: Centralized function to process and save files to Firestore
def process_and_save_files(db, user_email, files):
    with st.spinner("Processing and saving files..."):
        attachments_data = []
        cv_text = ""

        # Process CV first
        cv_file = files[0]
        cv_bytes = cv_file.getvalue()
        attachments_data.append({
            "name": cv_file.name,
            "content_b64": base64.b64encode(cv_bytes).decode('utf-8')
        })
        
        # Extract text from CV
        raw_cv_text = extract_text_from_pdf(cv_bytes)
        if raw_cv_text:
            # Simple language check
            if any(char in 'àéâç' for char in raw_cv_text[:500]): # Basic check for French/German
                 cv_text = raw_cv_text
            else: # Assume English, needs translation
                 cv_text = translate_cv_text(raw_cv_text)

        # Process other files
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
        # Reload state from DB
        st.session_state.attachments = attachments_data
        st.session_state.cv_text = cv_text
        st.rerun()

# ================================
# MAIN APP LAYOUT & LOGIC (HEAVILY MODIFIED)
# ================================
st.set_page_config(layout="wide")
st.title("🇨🇭 Swiss Assistenzarzt Job Application Bot")

# Initialize session state keys
default_states = {
    'user_email': None, 'gmail_service': None, 'cv_text': None,
    'attachments': [], 'current_job_id': None, 'db': None,
    'generated_email_content': None, 'manual_email_content': None,
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Connect to DB once
if not st.session_state.db:
    st.session_state.db = get_firestore_db()
if not st.session_state.db:
    st.error("Could not connect to the database. The app cannot continue.")
    st.stop()


# --- Primary App Flow ---
if not st.session_state.user_email:
    st.header("Step 1: Authorize Your Gmail Account")
    st.info("This app needs permission to send emails on your behalf and view your email address to create your profile.")
    if st.button("Login with Google"):
        service, email = gmail_authenticate()
        if service and email:
            st.session_state.gmail_service = service
            st.session_state.user_email = email
            st.success(f"Logged in as {email}")
            st.rerun()
else:
    # User is logged in, load their data
    user_data = get_user_data(st.session_state.db, st.session_state.user_email)
    st.session_state.cv_text = user_data.get("cv_text")
    st.session_state.attachments = user_data.get("attachments", [])

    # Check if user needs to upload a CV
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
        # Main App Interface
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


            