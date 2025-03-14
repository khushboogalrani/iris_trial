import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import google.auth.transport.requests

# Define the SCOPES for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Path to store the token.json (if not using the default location)
TOKEN_PATH = 'token.json'

def authenticate_gmail_api():
    """Authenticate with Gmail API using OAuth2."""
    # Get credentials from GitHub Secret
    credentials_json = os.getenv('GMAIL_CREDENTIALS_JSON')

    if credentials_json is None:
        print("No credentials JSON found in environment variables.")
        exit(1)
    
    credentials = None
    # The credentials file will be stored temporarily in the environment
    with open('credentials.json', 'w') as f:
        f.write(credentials_json)

    # The flow is used to authenticate and obtain the credentials
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)

    credentials = flow.run_local_server(port=0)

    # Save the credentials for the next run
    with open(TOKEN_PATH, 'w') as token:
        token.write(credentials.to_json())

    # Build the Gmail API service
    service = build('gmail', 'v1', credentials=credentials)
    return service

def send_email():
    """Send an email using the Gmail API."""
    service = authenticate_gmail_api()

    # Create the email message
    message = MIMEMultipart()
    message['to'] = 'khushboo.krishn@gmai.com'
    message['subject'] = 'Test Email from GitHub Actions'

    msg = MIMEText('This is a test email sent via GitHub Actions and OAuth2 authentication.')
    message.attach(msg)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    # Send the email
    send_message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
    print('Message sent successfully: %s' % send_message['id'])

if __name__ == '__main__':
    send_email()
