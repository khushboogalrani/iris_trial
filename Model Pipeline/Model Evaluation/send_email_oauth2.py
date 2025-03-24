import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_email_notification():
    """Send email notification about pipeline status"""
    print("Preparing email notification")
    
    try:
        # Email configuration
        sender_email = os.environ.get("EMAIL_ADDRESS")
        receiver_email = os.environ.get("EMAIL_ADDRESS")
        app_password = os.environ.get("EMAIL_APP_PASSWORD")
        job_status = os.environ.get("JOB_STATUS", "Unknown")
        
        if not all([sender_email, receiver_email, app_password]):
            print("Email credentials not set in environment variables")
            return False
        
        # Get pipeline info from GitHub Actions
        github_repository = os.environ.get("GITHUB_REPOSITORY", "Unknown")
        github_workflow = os.environ.get("GITHUB_WORKFLOW", "AskNEU Pipeline")
        github_run_id = os.environ.get("GITHUB_RUN_ID", "Unknown")
        github_sha = os.environ.get("GITHUB_SHA", "Unknown")[:7]  # Short SHA
        
        # Create email content
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = f"AskNEU Pipeline Status: {job_status} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>AskNEU Pipeline Status Report</h2>
            <p><b>Status:</b> {job_status}</p>
            <p><b>Repository:</b> {github_repository}</p>
            <p><b>Workflow:</b> {github_workflow}</p>
            <p><b>Run ID:</b> {github_run_id}</p>
            <p><b>Commit:</b> {github_sha}</p>
            <p><b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Pipeline Steps:</h3>
            <ul>
                <li><b>Build:</b> {"Completed" if job_status == "success" else "Failed"}</li>
                <li><b>Deploy:</b> {"Completed" if job_status == "success" else "Failed"}</li>
            </ul>
            
            <p>Please check the GitHub Actions logs for more details.</p>
        </body>
        </html>
        """
        
        message.attach(MIMEText(body, "html"))
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(message)
            
        print("Email notification sent successfully")
        return True
    except Exception as e:
        print(f"Error sending email notification: {str(e)}")
        return False

if __name__ == "__main__":
    send_email_notification()
