import yagmail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path

def yagmail_results(path, msg, alert_config):

    if alert_config['suppress_alerts']:
        return
    if not os.path.exists(path):
        return
    yag = yagmail.SMTP(oauth2_file = alert_config['path_to_twilio_secret'])
    yag.send(alert_config['alert_emails'], 'Genetic Vars Summary', [msg], attachments = path)