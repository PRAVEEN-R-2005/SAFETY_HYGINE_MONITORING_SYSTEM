import os
import smtplib
import threading
import queue
import requests
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from src.utils import logger
from config import Config

# Global thread-safe queue for real-time dashboard notifications
ui_alert_queue = queue.Queue(maxsize=100)

class AlertSystem:
    def __init__(self, db_session=None):
        self.db_session = db_session
        
    def dispatch_alerts(self, violation, screenshot_path=None):
        """
        Dispatches all enabled alerts asynchronously in background threads.
        """
        # 1. Real-time UI Alerts (Queue-based, consumed by SSE stream)
        try:
            if ui_alert_queue.full():
                try:
                    ui_alert_queue.get_nowait() # Remove oldest
                except queue.Empty:
                    pass
            ui_alert_queue.put_nowait({
                'id': violation.id if hasattr(violation, 'id') else 0,
                'timestamp': datetime_to_str(violation.timestamp) if hasattr(violation, 'timestamp') else '',
                'violation_type': violation.violation_type,
                'camera_name': violation.camera.name if hasattr(violation, 'camera') else f"Cam {violation.camera_id}",
                'confidence': violation.confidence,
                'screenshot_url': f"/static/screenshots/{os.path.basename(screenshot_path)}" if screenshot_path else '',
                'sound_alarm': True
            })
        except Exception as e:
            logger.error(f"Error putting alert in UI queue: {e}")

        # Check configuration flags
        email_enabled = Config.SMTP_USER != '' and Config.ALERT_EMAIL_TO != ''
        whatsapp_enabled = Config.TWILIO_ACCOUNT_SID != '' and Config.ALERT_WHATSAPP_TO != ''

        # 2. Email alert thread
        if email_enabled:
            threading.Thread(
                target=self._send_email_worker,
                args=(violation, screenshot_path),
                daemon=True
            ).start()
        else:
            logger.debug("Email alerts skipped: SMTP configs not set.")
            
        # 3. WhatsApp alert thread
        if whatsapp_enabled:
            threading.Thread(
                target=self._send_whatsapp_worker,
                args=(violation),
                daemon=True
            ).start()
        else:
            logger.debug("WhatsApp alerts skipped: Twilio credentials not set.")

    def _send_email_worker(self, violation, screenshot_path):
        """Worker thread to run SMTP transactions without slowing down video inference"""
        try:
            msg = MIMEMultipart('related')
            msg['Subject'] = f"[ALERT] Safety Violation: {violation.violation_type} detected!"
            msg['From'] = Config.SMTP_USER
            msg['To'] = Config.ALERT_EMAIL_TO

            # Build HTML body
            camera_name = violation.camera.name if hasattr(violation, 'camera') else f"Camera #{violation.camera_id}"
            time_str = violation.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(violation, 'timestamp') else 'Now'
            
            html = f"""
            <html>
              <head></head>
              <body style="font-family: Arial, sans-serif; background-color: #f6f6f6; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; border: 1px solid #ddd; overflow: hidden;">
                  <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center; font-size: 24px; font-weight: bold;">
                    SAFETY MONITORS ALERT
                  </div>
                  <div style="padding: 20px; color: #333;">
                    <p>Hello,</p>
                    <p>A safety violation has been detected in real time by the Safety & Hygiene Monitoring System.</p>
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                      <tr style="background: #f9f9f9; border-bottom: 1px solid #eee;">
                        <td style="padding: 8px; font-weight: bold; width: 150px;">Violation:</td>
                        <td style="padding: 8px; color: #dc3545; font-weight: bold;">{violation.violation_type}</td>
                      </tr>
                      <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 8px; font-weight: bold;">Camera:</td>
                        <td style="padding: 8px;">{camera_name}</td>
                      </tr>
                      <tr style="background: #f9f9f9; border-bottom: 1px solid #eee;">
                        <td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                        <td style="padding: 8px;">{time_str}</td>
                      </tr>
                      <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 8px; font-weight: bold;">Confidence:</td>
                        <td style="padding: 8px;">{violation.confidence * 100:.1f}%</td>
                      </tr>
                    </table>
                    <p style="text-align: center; margin-top: 25px;">
                      <img src="cid:screenshot" style="max-width: 100%; border-radius: 4px; border: 2px solid #333;" alt="Violation Screenshot" />
                    </p>
                    <p style="margin-top: 20px; font-size: 12px; color: #777; text-align: center;">
                      This is an automated message. Please check the dashboard to mark this issue as resolved.
                    </p>
                  </div>
                </div>
              </body>
            </html>
            """
            
            msgAlternative = MIMEMultipart('alternative')
            msg.attach(msgAlternative)
            msgText = MIMEText(html, 'html')
            msgAlternative.attach(msgText)

            # Attach Image
            if screenshot_path and os.path.exists(screenshot_path):
                with open(screenshot_path, 'rb') as fp:
                    msgImage = MIMEImage(fp.read())
                msgImage.add_header('Content-ID', '<screenshot>')
                msg.attach(msgImage)

            # Connect SMTP server
            if Config.SMTP_PORT == 465:
                server = smtplib.SMTP_SSL(Config.SMTP_SERVER, Config.SMTP_PORT)
            else:
                server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
                server.starttls()
                
            server.login(Config.SMTP_USER, Config.SMTP_PASS)
            server.sendmail(Config.SMTP_USER, Config.ALERT_EMAIL_TO, msg.as_string())
            server.quit()
            logger.info(f"Email violation alert sent to {Config.ALERT_EMAIL_TO} for violation: {violation.violation_type}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_whatsapp_worker(self, violation):
        """Worker thread to dispatch Twilio API WhatsApp messages"""
        try:
            camera_name = violation.camera.name if hasattr(violation, 'camera') else f"Camera #{violation.camera_id}"
            time_str = violation.timestamp.strftime("%H:%M:%S") if hasattr(violation, 'timestamp') else 'Now'
            text_body = (
                f"*SAFETY INCIDENT DETECTED*\n"
                f"• *Violation:* {violation.violation_type}\n"
                f"• *Source:* {camera_name}\n"
                f"• *Time:* {time_str}\n"
                f"• *Confidence:* {violation.confidence * 100:.1f}%\n"
                f"Please inspect the monitoring panel dashboard immediately."
            )
            
            url = f"https://api.twilio.com/2010-04-01/Accounts/{Config.TWILIO_ACCOUNT_SID}/Messages.json"
            payload = {
                'From': Config.TWILIO_WHATSAPP_FROM,
                'To': Config.ALERT_WHATSAPP_TO,
                'Body': text_body
            }
            
            # Twilio Auth
            auth = (Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
            response = requests.post(url, data=payload, auth=auth)
            
            if response.status_code in [200, 201]:
                logger.info(f"WhatsApp alert successfully sent to {Config.ALERT_WHATSAPP_TO}")
            else:
                logger.error(f"WhatsApp alert via Twilio failed. Code: {response.status_code}, Body: {response.text}")
        except Exception as e:
            logger.error(f"Error dispatching WhatsApp notification: {e}")

def datetime_to_str(dt):
    if isinstance(dt, datetime.datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)
