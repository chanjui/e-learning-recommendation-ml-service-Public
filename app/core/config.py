from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "E-Learning ML Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React 프론트엔드
        "http://localhost:8080",  # Spring Boot 백엔드
        "*"  # 개발 환경에서만 사용
    ]
    
    # OAuth2 설정
    GOOGLE_CLIENT_ID: str = "617172854156-b05pg0gp7ktjl08lcil6svh8bj87dnvl.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI: str = "http://localhost:3000/auth/auth/google-callback"
    
    KAKAO_CLIENT_ID: str = "e3aecedb65b63e1fac9d777cd66fb3e1"
    KAKAO_CLIENT_SECRET: str = os.getenv("KAKAO_CLIENT_SECRET")
    KAKAO_REDIRECT_URI: str = "http://localhost:3000/auth/auth/kakao-callback"
    
    GITHUB_CLIENT_ID: str = "Ov23lihKiKICvlydhVsj"
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI: str = "http://localhost:3000/auth/auth/github-callback"

settings = Settings() 