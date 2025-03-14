import os
import base64
import logging
import argparse
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the SCOPES for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def authenticate_gmail_api():
    """Authenticate the Gmail API using OAuth2"""
    creds = None
    if os.path.exists('credentials.json'):
        creds = Credentials.from_authorized_user_file('credentials.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('credentials.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def send_email(subject, message):
    """Send an email via Gmail API"""
    service = authenticate_gmail_api()
    
    message_obj = MIMEMultipart()
    message_obj['to'] = 'khushboo.krishna@gmail.com'  # Replace with the recipient email
    message_obj['subject'] = subject
    message_obj.attach(MIMEText(message, 'plain'))

    raw_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
    
    try:
        message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        logging.info(f"Message sent: {message['id']}")
    except Exception as error:
        logging.error(f"An error occurred: {error}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Send an email using Gmail API")
    parser.add_argument('--subject', required=True, help="Subject of the email")
    parser.add_argument('--message', required=True, help="Message content of the email")

    args = parser.parse_args()

    send_email(args.subject, args.message)
