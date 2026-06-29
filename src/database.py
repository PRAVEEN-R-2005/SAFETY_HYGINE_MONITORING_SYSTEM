import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config import Config

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='operator') # admin, manager, operator
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Required for flask-login
    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    employee_code = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(50), nullable=True)
    status = Column(String(20), default='active') # active, inactive
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    stream_url = Column(String(255), nullable=False) # webcam index (0), RTSP url, or file path
    department = Column(String(50), nullable=True)
    status = Column(String(20), default='active') # active, inactive
    restricted_area_coords = Column(Text, nullable=True) # JSON string of coordinates: [[x1,y1], [x2,y2], ...]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_active = Column(DateTime, nullable=True)

    violations = relationship("Violation", back_populates="camera", cascade="all, delete-orphan")
    detection_logs = relationship("DetectionLog", back_populates="camera", cascade="all, delete-orphan")

    def get_polygon(self):
        if not self.restricted_area_coords:
            return []
        try:
            return json.loads(self.restricted_area_coords)
        except Exception:
            return []

    def set_polygon(self, coords_list):
        self.restricted_area_coords = json.dumps(coords_list)


class Violation(Base):
    __tablename__ = 'violations'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    violation_type = Column(String(100), nullable=False) # e.g., 'No Helmet', 'No Vest', 'Unauthorized Access', 'Fire'
    confidence = Column(Float, nullable=False)
    screenshot_path = Column(String(255), nullable=True)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)

    camera = relationship("Camera", back_populates="violations")
    alerts = relationship("AlertLog", back_populates="violation", cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    report_type = Column(String(20), nullable=False) # 'PDF', 'Excel'
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    file_path = Column(String(255), nullable=False)
    status = Column(String(20), default='completed') # completed, failed


class AlertLog(Base):
    __tablename__ = 'alert_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    violation_id = Column(Integer, ForeignKey('violations.id'), nullable=False)
    alert_type = Column(String(50), nullable=False) # 'Email', 'WhatsApp', 'Sound', 'Popup'
    status = Column(String(20), default='sent') # sent, failed
    error_message = Column(Text, nullable=True)

    violation = relationship("Violation", back_populates="alerts")


class Setting(Base):
    __tablename__ = 'settings'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String(255), nullable=True)


class DetectionLog(Base):
    __tablename__ = 'detection_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    worker_count = Column(Integer, default=0)
    compliance_percentage = Column(Float, default=100.0) # Compliance = (Workers with all PPE / Total Workers) * 100
    ppe_stats_json = Column(Text, nullable=True) # JSON store of counts: {"helmets": X, "vests": Y, "gloves": Z}

    camera = relationship("Camera", back_populates="detection_logs")


# Database engine and session creator
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_base_all = Base.metadata.create_all(engine)
    
    # Create default settings if not exists
    db_session = SessionLocal()
    try:
        # Check if default admin exists
        admin = db_session.query(User).filter_by(username='admin').first()
        if not admin:
            import bcrypt
            hashed = bcrypt.hashpw(b'admin123', bcrypt.gensalt()).decode('utf-8')
            default_admin = User(username='admin', password_hash=hashed, role='admin')
            db_session.add(default_admin)
            
        # Add default settings
        defaults = {
            'email_alerts_enabled': 'true',
            'whatsapp_alerts_enabled': 'false',
            'sound_alerts_enabled': 'true',
            'popup_alerts_enabled': 'true',
            'detection_interval_ms': '500', # process every 500ms
            'violation_cooldown_seconds': '300', # 5 mins
        }
        for k, v in defaults.items():
            setting = db_session.query(Setting).filter_by(key=k).first()
            if not setting:
                db_session.add(Setting(key=k, value=v, description=f"Default setting for {k}"))
        db_session.commit()
    except Exception as e:
        print(f"Error initializing DB: {e}")
        db_session.rollback()
    finally:
        db_session.close()

def get_db():
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
