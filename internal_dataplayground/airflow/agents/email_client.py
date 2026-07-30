# airflow/agents/email_client.py
"""
Generic Gmail SMTP sender for Airflow DAGs.

Deliberately not job-specific — meant to be the one place any future
"newsletter of things that matter" digest sends mail from (jobs today;
habits/journal/blog pipeline status later), so you don't end up with
several different smtplib call sites over time.

SETUP (one-time, ~5 minutes):
  1. Turn on 2-Step Verification on the Gmail account you want to send
     from (Google Account -> Security). Required before Google will issue
     App Passwords — this works fine on a free personal Gmail account,
     nothing Workspace-specific about it.
  2. Google Account -> Security -> App Passwords -> create one named
     something like "life-os-digest". Copy the 16-character password.
  3. Store it as GMAIL_APP_PASSWORD and the sending/receiving address as
     GMAIL_ADDRESS — same env-var pattern as GEMINI_API etc. in
     docker-compose.yml. Do NOT use your actual Google account password.

Gmail's SMTP relay caps free personal accounts around 500 sends/day. A
once-daily email to yourself is nowhere near that — no local mail server
or third-party transactional email API is needed for this use case. Those
only start to matter if you're sending high volume or to many distinct
recipients, neither of which applies here.
"""
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_email(subject: str, html_body: str, to_address: str | None = None) -> None:
    """
    Sends an HTML email via Gmail's SMTP relay.

    Args:
        subject:    Email subject line.
        html_body:  Full HTML content. Email clients need inline styles —
                    don't rely on a <style> block or external CSS.
        to_address: Defaults to GMAIL_ADDRESS (mail yourself) if omitted.
    """
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_address or not app_password:
        raise RuntimeError(
            "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — see the setup "
            "instructions in this file's docstring."
        )

    recipient = to_address or gmail_address

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(gmail_address, app_password)
        server.sendmail(gmail_address, [recipient], msg.as_string())

    log.info("Sent email '%s' to %s", subject, recipient)
