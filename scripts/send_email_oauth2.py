import os.path
import base64
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying the email, ensure the scopes match
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Gmail API Authentication
def authenticate_gmail_api():
    creds = None
    # The file token.json stores the user's access and refresh tokens and is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

# Send email using Gmail API
def send_email():
    try:
        service = authenticate_gmail_api()

        message = MIMEMultipart()
        message['to'] = "khushboo.krishna@gmail.com"  # Replace with the recipient email
        message['from'] = "khushboo.krishna@gmail.com"  # Your Gmail address
        message['subject'] = "Test Email from Gmail API using OAuth 2.0"
        
        # The body of the email
        body = "This is a test email sent from Python using OAuth 2.0 authentication."
        message.attach(MIMEText(body, 'plain'))

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Send the email
        message = service.users().messages().send(userId="me", body={'raw': raw_message}).execute()
        print(f"Message sent successfully, Message ID: {message['id']}")
    except HttpError as error:
        print(f'An error occurred: {error}')

# Run the function
send_email()
