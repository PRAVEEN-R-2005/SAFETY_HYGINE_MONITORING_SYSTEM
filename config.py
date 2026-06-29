import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'safety-hygiene-super-secret-key-12345')
    
    # Project paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    SCREENSHOTS_FOLDER = os.path.join(BASE_DIR, 'static', 'screenshots')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
    LOGS_FOLDER = os.path.join(BASE_DIR, 'logs')
    
    # Create required folders
    for folder in [UPLOAD_FOLDER, SCREENSHOTS_FOLDER, REPORTS_FOLDER, LOGS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        
    # Database configuration
    # By default, SQLite. Can be overridden with MySQL URI: mysql+pymysql://user:pass@host/dbname
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        f"sqlite:///{os.path.join(BASE_DIR, 'database', 'safety_monitor.db')}"
    )
    # Ensure database folder exists
    os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI/Detection parameters
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.45))
    IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.45))
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'best.pt'))
    
    # Alert configurations
    # Email settings (SMTP)
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USER = os.environ.get('SMTP_USER', '')
    SMTP_PASS = os.environ.get('SMTP_PASS', '')
    ALERT_EMAIL_TO = os.environ.get('ALERT_EMAIL_TO', '')
    
    # WhatsApp (Twilio)
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
    TWILIO_WHATSAPP_FROM = os.environ.get('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886') # Twilio sandbox default
    ALERT_WHATSAPP_TO = os.environ.get('ALERT_WHATSAPP_TO', '')
    
    # System settings defaults
    ALERT_COOLDOWN_SECONDS = int(os.environ.get('ALERT_COOLDOWN_SECONDS', 300)) # 5 minutes per unique worker/violation
    ALARM_SOUND_ENABLED = True
    POPUP_ALERTS_ENABLED = True
