from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.services.recommendation_service import RecommendationService
from typing import List, Dict, Any
from app.api.endpoints import auth
import logging
from pydantic import BaseModel

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])

class CourseRecommendation(BaseModel):
    id: int
    title: str
    description: str
    price: float
    thumbnail_url: str | None = None
    rating: float
    category: str
    instructor_name: str
    enrollment_count: int

class InstructorRecommendation(BaseModel):
    id: int
    name: str
    bio: str | None = None
    profile_url: str | None = None
    specialization: str | None = None
    rating: float
    course_count: int
    total_enrollments: int

class RecommendationResponse(BaseModel):
    courses: List[CourseRecommendation]
    instructors: List[InstructorRecommendation]

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    db: Session = Depends(get_db)
) -> RecommendationResponse:
    """
    사용자에게 맞는 강의와 강사 추천
    
    - 강의 추천 기준:
        1. 사용자가 아직 수강하지 않은 강의
        2. 평점이 높은 순
        3. 수강생이 많은 순
        
    - 강사 추천 기준:
        1. 사용자가 아직 수강하지 않은 강사의 강의
        2. 강사 평점이 높은 순
        3. 총 수강생이 많은 순
    """
    try:
        logger.info(f"Getting recommendations for user {user_id}")
        service = RecommendationService(db)
        
        # 강의 추천
        course_recommendations = service.get_course_recommendations(user_id)
        logger.info(f"Found {len(course_recommendations)} course recommendations")
        
        # 강사 추천
        instructor_recommendations = service.get_instructor_recommendations(user_id)
        logger.info(f"Found {len(instructor_recommendations)} instructor recommendations")
        
        # 강의 추천 결과를 CourseRecommendation 모델로 변환
        courses = [
            CourseRecommendation(
                id=course["id"],
                title=course["title"],
                description=course["description"],
                price=course["price"],
                thumbnail_url=course.get("thumbnail_url"),
                rating=course["rating"],
                category=course["category"],
                instructor_name=course["instructor_name"],
                enrollment_count=course["enrollment_count"]
            ) for course in course_recommendations
        ]
        
        # 강사 추천 결과를 InstructorRecommendation 모델로 변환
        instructors = [
            InstructorRecommendation(
                id=instructor["id"],
                name=instructor["name"],
                bio=instructor.get("bio"),
                profile_url=instructor.get("profile_url"),
                specialization=instructor.get("specialization"),
                rating=instructor["rating"],
                course_count=instructor["course_count"],
                total_enrollments=instructor["total_enrollments"]
            ) for instructor in instructor_recommendations
        ]
        
        return RecommendationResponse(courses=courses, instructors=instructors)
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 시스템 오류: {str(e)}"
        ) 