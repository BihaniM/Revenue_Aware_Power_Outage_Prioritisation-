from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, UniqueConstraint
from app.database import Base

class User(Base):
    __tablename__ = "login_details"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, index=True, nullable=False)
    password_plain = Column(String(255), nullable=False)
    role = Column(String(30), default="common", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SelectedSite(Base):
    __tablename__ = "selected_sites"
    id = Column(Integer, primary_key=True, index=True)
    site_name = Column(String(255), unique=True, index=True, nullable=False)
    region = Column(String(80), index=True, nullable=False)

class KpiData(Base):
    __tablename__ = "kpi_data"
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, index=True, nullable=False)
    period_min = Column(Integer, default=60)
    site_name = Column(String(255), index=True, nullable=False)
    data_traffic_mb = Column(Float, default=0.0)
    voice_traffic_erl = Column(Float, default=0.0)
    hour_of_day = Column(Integer, index=True)
    import_tag = Column(String(80), default="training")
    created_at = Column(DateTime, default=datetime.utcnow)

class PowerAlarm(Base):
    __tablename__ = "power_alarms"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(80), index=True, nullable=False)
    site_name = Column(String(255), index=True, nullable=False)
    region = Column(String(80), index=True)
    event_date = Column(Date, index=True)
    alarm_sequence = Column(Integer)
    alarm_name = Column(String(255), index=True, nullable=False)
    occurred_on = Column(DateTime, index=True, nullable=False)
    cleared_on = Column(DateTime)
    alarm_duration_min = Column(Float, default=0.0)
    import_tag = Column(String(80), default="training")
    created_at = Column(DateTime, default=datetime.utcnow)

class SiteAlarmAverage(Base):
    __tablename__ = "average_alarm_durations"
    id = Column(Integer, primary_key=True, index=True)
    site_name = Column(String(255), unique=True, index=True, nullable=False)
    region = Column(String(80), index=True)
    avg_mains_to_dc_min = Column(Float, default=0.0)
    avg_dc_to_down_min = Column(Float, default=0.0)
    avg_site_down_duration_min = Column(Float, default=0.0)
    sample_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)

class RevenuePriority(Base):
    __tablename__ = "dynamic_prioritized_list"
    id = Column(Integer, primary_key=True, index=True)
    site_name = Column(String(255), unique=True, index=True, nullable=False)
    region = Column(String(80), index=True)
    hour_of_day = Column(Integer, index=True)
    avg_data_mb_per_hour = Column(Float, default=0.0)
    avg_voice_erl_per_hour = Column(Float, default=0.0)
    data_revenue_hour = Column(Float, default=0.0)
    voice_revenue_hour = Column(Float, default=0.0)
    total_revenue_hour = Column(Float, default=0.0)
    priority_rank = Column(Integer, index=True)
    alarm_state = Column(String(50), default="Normal")
    last_alarm_time = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "prediction_results"
    id = Column(Integer, primary_key=True, index=True)
    site_name = Column(String(255), index=True, nullable=False)
    region = Column(String(80), index=True)
    alarm_state = Column(String(50), default="Power failure")
    event_time = Column(DateTime, index=True)
    predicted_mains_to_dc_min = Column(Float, default=0.0)
    predicted_dc_to_down_min = Column(Float, default=0.0)
    predicted_site_down_duration_min = Column(Float, default=0.0)
    data_loss_1h = Column(Float, default=0.0)
    data_loss_2h = Column(Float, default=0.0)
    data_loss_3h = Column(Float, default=0.0)
    data_loss_4h = Column(Float, default=0.0)
    voice_loss_1h = Column(Float, default=0.0)
    voice_loss_2h = Column(Float, default=0.0)
    voice_loss_3h = Column(Float, default=0.0)
    voice_loss_4h = Column(Float, default=0.0)
    total_loss_1h = Column(Float, default=0.0)
    total_loss_2h = Column(Float, default=0.0)
    total_loss_3h = Column(Float, default=0.0)
    total_loss_4h = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class ImportLog(Base):
    __tablename__ = "import_logs"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255))
    import_type = Column(String(40))
    import_tag = Column(String(80))
    row_count = Column(Integer, default=0)
    message = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
