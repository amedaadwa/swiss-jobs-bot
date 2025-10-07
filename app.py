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
from openai import OpenAI  # <-- NEW: OpenAI import

# ================================
# SETUP & CONFIGURATION
# ================================
load_dotenv()
# --- NEW: Using OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

CSV_FILE = "Assistenzarzt_Jobs_CH__Combined_Final.csv"
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
DB_COLLECTION = "job_applications_v2"
DB_DOCUMENT_ID = "user_profile"

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
    db = get_firestore_db()
    creds = None
    if db:
        doc_ref = db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID)
        doc = doc_ref.get()
        if doc.exists and 'gmail_token' in doc.to_dict():
            creds = pickle.loads(base64.b64decode(doc.to_dict()['gmail_token']))

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
            if not client_secret_json:
                st.error("Google client secret JSON not found.")
                return None
            client_config = json.loads(client_secret_json)
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)
        
        if db:
            token_b64 = base64.b64encode(pickle.dumps(creds)).decode('utf-8')
            doc_ref.set({'gmail_token': token_b64}, merge=True)
            
    return build("gmail", "v1", credentials=creds)

# ================================
# API & HELPER FUNCTIONS
# ================================
def call_openai_api(prompt, system_message="You are a helpful assistant."):
    """Generic function to call the OpenAI API."""
    try:
        if not OPENAI_API_KEY:
            st.error("OpenAI API key is not configured. Please add it to your environment variables.")
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

def generate_personalized_email(job_title, hospital_name, canton, job_description, cv_content):
    """Generates an email subject and body using OpenAI."""
    prompt = f"""
Act as a professional medical career advisor in Switzerland.
Your task is to create a compelling application email in German.

**Applicant's Profile (from CV):**
---
{cv_content}
---
**Job Details:**
- Position: {job_title}
- Hospital: {hospital_name}
- Canton: {canton}
- Job Description:
---
{job_description}
---

**Instructions:**
1.  **Generate a Subject Line:** Create a concise, professional subject line in German.
2.  **Generate an Email Body:** Write a polite, personalized email connecting the applicant's CV to the job description.
3.  **Output Format:** Your final output MUST contain the subject and body separated by '|||'.
    Example: Betreff: Bewerbung als Assistenzarzt|||Sehr geehrte Damen und Herren,...
"""
    response = call_openai_api(prompt, "You are a professional medical job applicant assistant, writing in German.")
    if response and '|||' in response:
        parts = response.split('|||', 1)
        subject = parts[0].replace('Betreff:', '').replace('Subject:', '').strip()
        body = parts[1].strip()
        return {'subject': subject, 'body': body}
    
    return {'subject': f"Bewerbung als {job_title}", 'body': response or "Could not generate email body."}

def translate_cv_text(text):
    prompt = f"Please translate the following CV text from English to professional, high-quality German suitable for a medical job application in Switzerland.\n\n**Text to Translate:**\n---\n{text}\n---"
    system_message = "You are an expert translator specializing in medical and professional documents."
    return call_openai_api(prompt, system_message)

def send_email_logic(service, to_email, subject, body, attachments):
    try:
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to_email
        message["Subject"] = subject
        message["From"] = "me"
        for uploaded_file in attachments:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            message.add_attachment(content, maintype="application", subtype="octet-stream", filename=uploaded_file.name)
        encoded_message = base64.b64encode(message.as_bytes()).decode()
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
def get_user_data(db):
    if not db: return {}
    doc = db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).get()
    return doc.to_dict() if doc.exists else {}

def update_user_data(db, data_to_update):
    if not db: return
    db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).set(data_to_update, merge=True)

def save_sent_email(db, email_data):
    if not db: return
    db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).collection("sent_emails").add(email_data)

# ================================
# UI PAGE FUNCTIONS
# ================================
def render_dashboard(db):
    st.header("📊 Dashboard: Sent Applications")
    emails_ref = db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).collection("sent_emails").order_by("sent_at", direction=firestore.Query.DESCENDING).stream()
    
    emails = list(emails_ref)
    if not emails:
        st.info("You haven't sent any emails yet. Head over to the 'Job Finder' to get started!")
        return

    for email in emails:
        data = email.to_dict()
        sent_time = data['sent_at'].strftime("%d %b %Y, %H:%M")
        with st.expander(f"To: {data['recipient']} | Subject: {data['subject']} | Sent: {sent_time}"):
            st.write(f"**To:** {data['recipient']}")
            st.write(f"**Subject:** {data['subject']}")
            st.write(f"**Sent At:** {sent_time}")
            st.markdown("---")
            st.text_area("Email Body", value=data['body'], height=300, disabled=True, key=f"body_{email.id}")

def render_manual_job_page(db):
    st.header("✍️ Add a Job Manually")
    st.info("Fill in the details for a job that isn't in the CSV list.")

    with st.form("manual_job_form"):
        job_title = st.text_input("Job Title")
        hospital_name = st.text_input("Hospital Name")
        canton = st.text_input("Canton")
        contact_email = st.text_input("Contact Email")
        job_description = st.text_area("Job Description", height=150)
        submitted = st.form_submit_button("Generate Email for this Job")

    if submitted and all([job_title, hospital_name, contact_email]):
        st.session_state.manual_job_details = {
            "job_title": job_title, "hospital_name": hospital_name, "canton": canton,
            "contact_email": contact_email, "job_description": job_description
        }
        with st.spinner("Generating email with OpenAI..."):
            email_content = generate_personalized_email(
                job_title, hospital_name, canton, job_description, st.session_state.cv_content
            )
            st.session_state.manual_email_content = email_content
    elif submitted:
        st.warning("Please fill in at least the Job Title, Hospital, and Email.")

    if 'manual_email_content' in st.session_state and st.session_state.manual_email_content:
        render_application_form(db, is_manual=True)

def render_job_finder(db):
    st.header("🔍 Job Finder")
    try:
        jobs_df = pd.read_csv(CSV_FILE)
        user_data = get_user_data(db)
        applied_jobs = set(user_data.get("applied_jobs", []))
    except FileNotFoundError:
        st.error(f"Error: '{CSV_FILE}' not found.")
        st.stop()
    
    next_job = None
    for idx, row in jobs_df.iterrows():
        if str(idx) not in applied_jobs and isinstance(row.get("Application Contact Email"), str) and "@" in row.get("Application Contact Email", ""):
            next_job = (idx, row)
            break
            
    if next_job is None:
        st.info("🎉 All jobs from the CSV have been processed!")
        return

    job_id, row_data = next_job
    
    if st.session_state.current_job_id != job_id:
        st.session_state.current_job_id = job_id
        st.session_state.generated_email_content = None
        st.session_state.current_job_details = {
            "job_id": job_id, "job_title": row_data.get("job_title", "N/A"),
            "hospital_name": row_data.get("hospital_name", ""), "canton": row_data.get("canton", ""),
            "contact_email": str(row_data.get("Application Contact Email", "")).strip().split(",")[0],
            "application_url": row_data.get("Application URL", ""),
            "job_description": row_data.get("Job Description (short)", "")
        }

    details = st.session_state.current_job_details
    st.subheader(f"Next Up: {details['job_title']}")
    st.write(f"**Hospital:** {details['hospital_name']} ({details['canton']})")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button(f"🤖 Prepare Application for Job #{details['job_id']}"):
             with st.spinner("Generating personalized email with OpenAI..."):
                email_content = generate_personalized_email(
                    details['job_title'], details['hospital_name'], details['canton'],
                    details['job_description'], st.session_state.cv_content
                )
                st.session_state.generated_email_content = email_content
    with col2:
        if st.button("Skip Job ⏭️"):
            update_user_data(db, {"applied_jobs": firestore.ArrayUnion([str(details['job_id'])])})
            db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).update({f'stats.skipped_count': firestore.Increment(1)})
            st.warning(f"Skipped job #{details['job_id']}. Moving to next.")
            st.session_state.current_job_id = None
            st.rerun()

    if st.session_state.get('generated_email_content'):
        render_application_form(db)

def render_application_form(db, is_manual=False):
    st.markdown("---")
    st.subheader("✉️ Review, Edit, and Send Application")

    if is_manual:
        details = st.session_state.manual_job_details
        email_content = st.session_state.manual_email_content
        form_key = "manual_form"
    else:
        details = st.session_state.current_job_details
        email_content = st.session_state.generated_email_content
        form_key = f"form_{details['job_id']}"

    with st.form(key=form_key):
        contact_email = st.text_input("To (Contact Email)", value=details.get('contact_email', ''))
        subject = st.text_input("Subject", value=email_content.get('subject', ''))
        body = st.text_area("Email Body", value=email_content.get('body', ''), height=350)
        
        st.markdown("**Attachments**")
        st.info("Manage attachments outside this form. New uploads will be added upon sending.")
        
        new_attachments = st.file_uploader("Add more files", accept_multiple_files=True, key=f"uploader_{form_key}")
        
        send_button = st.form_submit_button("🚀 Send Application")

    if st.session_state.attachments:
        st.write("Current Attachments:")
        # Loop backwards to avoid index errors when removing items
        for i in range(len(st.session_state.attachments) - 1, -1, -1):
            attached_file = st.session_state.attachments[i]
            c1, c2 = st.columns([0.8, 0.2])
            c1.info(f"📄 {attached_file.name}")
            if c2.button(f"Remove", key=f"remove_{i}_{form_key}_{attached_file.name}"):
                st.session_state.attachments.pop(i)
                st.rerun()

    if send_button:
        current_attachments = st.session_state.attachments + (new_attachments or [])
        if not current_attachments:
            st.warning("You must have at least one attachment (your CV)."); return

        if send_email_logic(st.session_state.gmail_service, contact_email, subject, body, current_attachments):
            email_record = {
                "recipient": contact_email, "subject": subject, "body": body,
                "sent_at": firestore.SERVER_TIMESTAMP, "job_title": details['job_title'],
                "hospital_name": details.get('hospital_name', 'Manual Entry')
            }
            save_sent_email(db, email_record)
            
            if not is_manual:
                update_user_data(db, {"applied_jobs": firestore.ArrayUnion([str(details['job_id'])])})
                db.collection(DB_COLLECTION).document(DB_DOCUMENT_ID).update({f'stats.sent_count': firestore.Increment(1)})
                st.session_state.current_job_id = None
            else:
                st.session_state.manual_email_content = None
            
            st.balloons()
            st.rerun()

# ================================
# MAIN APP LAYOUT & LOGIC
# ================================
st.set_page_config(layout="wide")
st.title("🇨🇭 Swiss Assistenzarzt Job Application Bot")

if 'step' not in st.session_state:
    st.session_state.step = "auth"
    st.session_state.gmail_service = None
    st.session_state.cv_content = None
    st.session_state.attachments = []
    st.session_state.current_job_id = None
    st.session_state.generated_email_content = None
    st.session_state.manual_email_content = None

db = get_firestore_db()
if not db: st.stop()

if st.session_state.step == "auth":
    st.header("Step 1: Authorize Your Gmail Account")
    if st.button("Authorize Gmail"):
        with st.spinner("Authenticating..."):
            st.session_state.gmail_service = gmail_authenticate()
        if st.session_state.gmail_service:
            st.session_state.step = "upload_cv"
            st.rerun()

elif st.session_state.step == "upload_cv":
    st.header("Step 2: Upload Your Documents")
    user_data = get_user_data(db)
    if user_data.get("translated_cv") and st.button("Use previously saved CV"):
        st.session_state.cv_content = user_data["translated_cv"]
        st.success("Loaded CV from database.")
        st.session_state.step = "main_app"
        st.rerun()

    uploaded_files = st.file_uploader("Upload your CV (must be the first file) and other attachments.", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.attachments = uploaded_files
        cv_file = uploaded_files[0]
        with st.spinner("Reading CV..."): cv_text = extract_text_from_pdf(cv_file)
        
        if cv_text:
            lang = st.radio("Is the CV in English (needs translation) or German?", ("English", "German"))
            if st.button("Confirm and Proceed"):
                if "English" in lang:
                    with st.spinner("Translating CV..."):
                        translated = translate_cv_text(cv_text)
                        st.session_state.cv_content = translated
                        update_user_data(db, {"translated_cv": translated})
                else:
                    st.session_state.cv_content = cv_text
                
                st.success("CV processed and saved!")
                st.session_state.step = "main_app"
                st.rerun()
            
elif st.session_state.step == "main_app":
    if not st.session_state.cv_content:
        # Attempt to load from DB one more time if not in state
        user_data = get_user_data(db)
        if user_data.get("translated_cv"):
             st.session_state.cv_content = user_data["translated_cv"]
        else:
            st.warning("CV has not been processed. Please return to the upload step.")
            if st.button("Go to Upload Step"):
                st.session_state.step = "upload_cv"
                st.rerun()
            st.stop()

    user_data = get_user_data(db)
    with st.sidebar:
        st.header("Navigation")
        app_page = st.radio("Go to", ["Job Finder", "Add Manual Job", "Dashboard"])
        
        st.markdown("---")
        stats = user_data.get("stats", {})
        st.header("📊 Statistics")
        st.metric("Emails Sent", stats.get("sent_count", 0))
        st.metric("Jobs Skipped", stats.get("skipped_count", 0))

    if app_page == "Job Finder": render_job_finder(db)
    elif app_page == "Dashboard": render_dashboard(db)
    elif app_page == "Add Manual Job": render_manual_job_page(db)

