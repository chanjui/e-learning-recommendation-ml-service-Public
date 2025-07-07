from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# MySQL 연결 설정
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# 엔진 생성 시 추가 옵션 설정
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=3600,   # 연결 재사용 시간 (1시간)
    pool_size=5,         # 커넥션 풀 크기
    max_overflow=10      # 최대 추가 커넥션 수
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 