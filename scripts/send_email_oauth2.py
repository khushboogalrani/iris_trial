import os
import argparse
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Define the SCOPES for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def authenticate_gmail_api():
    """Authenticate the Gmail API using OAuth2"""
    creds = None
    # Load credentials from environment variable (GMAIL_CREDENTIALS_JSON)
    credentials_json = os.getenv('GMAIL_CREDENTIALS_JSON')
    
    if credentials_json:
        creds = Credentials.from_authorized_user_info(json.loads(credentials_json), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(json.loads(credentials_json), SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('credentials.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def send_email(subject, message):
    """Send email using Gmail API"""
    service = authenticate_gmail_api()
    message = {
        'raw': create_message(subject, message)
    }
    send_message = service.users().messages().send(userId="me", body=message).execute()
    print(f"Message Id: {send_message['id']}")

def create_message(subject, message_body):
    """Create a message for the email"""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import base64

    message = MIMEMultipart()
    message['to'] = 'khushboo.krishna@example.com'  # Replace with actual recipient
    message['subject'] = subject
    message.attach(MIMEText(message_body, 'plain'))

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return raw_message

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', help='Subject of the email', required=True)
    parser.add_argument('--message', help='Body of the email', required=True)
    args = parser.parse_args()

    send_email(args.subject, args.message)
