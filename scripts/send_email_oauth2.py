import os
import json
import base64
import argparse
from email.mime.text import MIMEText  # Add this import for MIMEText
from google.auth.transport.requests import Request
from google.oauth2 import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Function to authenticate Gmail API using credentials
def authenticate_gmail_api():
    # Retrieve the Gmail credentials JSON from environment variable
    credentials_json = os.getenv('GMAIL_CREDENTIALS_JSON')
    print("Loaded credentials JSON:", credentials_json)
    
    if not credentials_json:
        raise ValueError("GMAIL_CREDENTIALS_JSON not found in environment variables.")
    
    # Parse the credentials JSON string into a dictionary
    try:
        credentials_dict = json.loads(credentials_json)
        print("Parsed credentials:", credentials_dict)
    except json.JSONDecodeError as e:
        print(f"Error parsing credentials JSON: {e}")
        raise

    creds = None
    try:
        # Try loading the credentials from the provided JSON string
        creds = Credentials.from_authorized_user_info(info=credentials_dict, scopes=SCOPES)
    except Exception as e:
        print(f"Error loading credentials: {e}")
        raise
    
    # If credentials are invalid, re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use the credentials from client configuration to start OAuth flow
            flow = InstalledAppFlow.from_client_config(credentials_dict, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use in 'credentials.json'
        with open('credentials.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

# Function to send the email
def send_email(subject, message):
    service = authenticate_gmail_api()
    
    # Replace the sender and recipient with actual email addresses
    message = create_message('khushboo.krishna@gmail.com', 'khushboo.krishna@gmail.com', subject, message)
    send_message(service, 'me', message)

# Function to create a message
def create_message(sender, to, subject, body):
    message = MIMEText(body)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

# Function to send the email
def send_message(service, sender, message):
    try:
        message = service.users().messages().send(userId=sender, body=message).execute()
        print(f"Message sent successfully: {message}")
        return message
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise

# Main function to parse arguments and send email
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', help='Subject of the email', required=True)
    parser.add_argument('--message', help='Message body of the email', required=True)
    args = parser.parse_args()
    
    send_email(args.subject, args.message)
