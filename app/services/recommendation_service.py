from app.models.course_recommender import CourseRecommender
from app.models.instructor_recommender import InstructorRecommender
from app.models.content_based_recommender import ContentBasedRecommender
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np
from typing import List, Dict, Any
import logging
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.course_recommender = CourseRecommender()
        self.instructor_recommender = InstructorRecommender()
        self.content_based_recommender = ContentBasedRecommender()
        
        # Load data and train the course recommender model on initialization
        logger.info("Initializing RecommendationService: Loading ratings and training model...")
        try:
            ratings_df = self._get_all_ratings()
            if not ratings_df.empty:
                self.course_recommender.fit(ratings_df)
            else:
                logger.warning("No ratings data found. Course recommender model is not trained.")
        except Exception as e:
            logger.error(f"Failed to initialize and train CourseRecommender: {e}", exc_info=True)
            # Depending on the desired behavior, you might want to handle this failure more gracefully

        # Load data and train the instructor recommender model on initialization
        logger.info("Initializing RecommendationService: Loading data and training INSTRUCTOR model...")
        try:
            instructor_avg_ratings_df = self._get_user_instructor_avg_ratings()
            if not instructor_avg_ratings_df.empty:
                self.instructor_recommender.fit(instructor_avg_ratings_df)
            else:
                logger.warning("No user-instructor avg ratings data found. Instructor recommender model is not trained.")
        except Exception as e:
            logger.error(f"Failed to initialize and train InstructorRecommender: {e}", exc_info=True)
            
        # Load data and train the content-based recommender model
        logger.info("Initializing RecommendationService: Loading course data and training content-based model...")
        try:
            courses_df = self._get_all_courses()
            if not courses_df.empty:
                self.content_based_recommender.fit(courses_df)
                logger.info("Content-based recommender model training complete.")
            else:
                logger.warning("No course data found. Content-based recommender model is not trained.")
        except Exception as e:
            logger.error(f"Failed to initialize and train ContentBasedRecommender: {e}", exc_info=True)

    def _get_all_ratings(self) -> pd.DataFrame:
        """Fetch all user-course ratings from the database."""
        try:
            query = text("SELECT userId, courseId, rating FROM courseRating WHERE rating IS NOT NULL")
            result = self.db.execute(query).fetchall()
            if not result:
                return pd.DataFrame(columns=['userId', 'courseId', 'rating'])

            # Convert to DataFrame
            df = pd.DataFrame(result, columns=['userId', 'courseId', 'rating'])
            # Ensure correct data types
            df['userId'] = df['userId'].astype(int)
            df['courseId'] = df['courseId'].astype(int)
            df['rating'] = df['rating'].astype(float) # Ratings might be integers
            logger.info(f"Loaded {len(df)} ratings into DataFrame.")
            return df
        except Exception as e:
            logger.error(f"Error fetching all ratings: {e}", exc_info=True)
            return pd.DataFrame(columns=['userId', 'courseId', 'rating']) # Return empty DataFrame on error

    def _get_user_instructor_avg_ratings(self) -> pd.DataFrame:
        """Fetch and calculate average rating given by each user to each instructor's courses."""
        try:
            # Query to get user, instructor, and rating for each rated course
            query = text("""
                SELECT
                    cr.userId,
                    c.instructorId,
                    cr.rating
                FROM courseRating cr
                JOIN course c ON cr.courseId = c.id
                WHERE cr.rating IS NOT NULL
            """)
            result = self.db.execute(query).fetchall()
            if not result:
                logger.warning("No course ratings found to calculate instructor average ratings.")
                return pd.DataFrame(columns=['userId', 'instructorId', 'avg_rating'])

            ratings_df = pd.DataFrame(result, columns=['userId', 'instructorId', 'rating'])
            # Calculate average rating per user per instructor
            avg_ratings_df = ratings_df.groupby(['userId', 'instructorId'])['rating'].mean().reset_index()
            avg_ratings_df.rename(columns={'rating': 'avg_rating'}, inplace=True)

            # Ensure correct types
            avg_ratings_df['userId'] = avg_ratings_df['userId'].astype(int)
            avg_ratings_df['instructorId'] = avg_ratings_df['instructorId'].astype(int)
            avg_ratings_df['avg_rating'] = avg_ratings_df['avg_rating'].astype(float)

            logger.info(f"Calculated {len(avg_ratings_df)} user-instructor average ratings.")
            return avg_ratings_df

        except Exception as e:
            logger.error(f"Error calculating user-instructor average ratings: {e}", exc_info=True)
            return pd.DataFrame(columns=['userId', 'instructorId', 'avg_rating'])

    def _get_all_active_course_ids(self) -> List[int]:
        """Fetch IDs of all active courses."""
        try:
            query = text("""
                SELECT c.id
                FROM course c
                JOIN instructor i ON c.instructorId = i.id
                JOIN `user` u ON i.userId = u.id
                WHERE c.status = 'ACTIVE' AND u.isDel = false
            """)
            result = self.db.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error fetching active course IDs: {e}", exc_info=True)
            return []

    def _get_user_enrolled_course_ids(self, user_id: int) -> List[int]:
        """Fetch IDs of courses the user is enrolled in."""
        try:
            query = text("SELECT courseId FROM courseEnrollment WHERE userId = :user_id")
            result = self.db.execute(query, {"user_id": user_id}).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error fetching enrolled courses for user {user_id}: {e}", exc_info=True)
            return []

    def _get_course_details_by_ids(self, course_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch detailed information for a list of course IDs."""
        if not course_ids:
            return []
        try:
            # Use IN clause for efficiency
            # Make sure placeholders match the number of IDs
            placeholders = ",".join([f":id_{i}" for i in range(len(course_ids))])
            params = {f"id_{i}": course_id for i, course_id in enumerate(course_ids)}

            query = text(f"""
                SELECT
                    c.id,
                    c.subject as title,
                    c.description,
                    c.price,
                    c.thumbnailUrl,
                    COALESCE(AVG(cr.rating), 0) as rating,
                    cat.name as category_name,
                    u.nickname as instructor_name,
                    COUNT(DISTINCT e.userId) as enrollment_count
                FROM course c
                JOIN instructor i ON c.instructorId = i.id
                JOIN `user` u ON i.userId = u.id
                JOIN category cat ON c.categoryId = cat.id
                LEFT JOIN courseEnrollment e ON c.id = e.courseId
                LEFT JOIN courseRating cr ON c.id = cr.courseId
                WHERE c.id IN ({placeholders})
                GROUP BY c.id, c.subject, c.description, c.price, c.thumbnailUrl, cat.name, u.nickname
            """)

            result = self.db.execute(query, params).fetchall()
            course_details = []
            for row in result:
                course_details.append({
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "price": float(row.price) if row.price is not None else 0.0,
                    "thumbnail_url": row.thumbnailUrl,
                    "rating": float(row.rating) if row.rating is not None else 0.0,
                    "category": row.category_name,
                    "instructor_name": row.instructor_name,
                    "enrollment_count": row.enrollment_count if row.enrollment_count is not None else 0
                })

            # Preserve the order of course_ids based on recommendation scores
            # Create a map for quick lookup
            details_map = {detail['id']: detail for detail in course_details}
            ordered_details = [details_map[cid] for cid in course_ids if cid in details_map]

            return ordered_details
        except Exception as e:
            logger.error(f"Error fetching course details for IDs {course_ids}: {e}", exc_info=True)
            return []

    def _get_all_active_instructor_ids(self) -> List[int]:
        """Fetch IDs of all active instructors."""
        try:
            query = text("SELECT i.id FROM instructor i JOIN `user` u ON i.userId = u.id WHERE u.isDel = false")
            result = self.db.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error fetching active instructor IDs: {e}", exc_info=True)
            return []

    def _get_instructors_user_interacted_with(self, user_id: int) -> List[int]:
        """Fetch IDs of instructors the user has interacted with (e.g., enrolled in their course)."""
        try:
            # Get instructors of courses the user enrolled in
            query = text("""
                SELECT DISTINCT c.instructorId
                FROM courseEnrollment ce
                JOIN course c ON ce.courseId = c.id
                WHERE ce.userId = :user_id
            """)
            result = self.db.execute(query, {"user_id": user_id}).fetchall()
            # We could also add instructors the user liked (likeTable type 2)
            # query_likes = text("SELECT targetUserId FROM likeTable WHERE userId = :user_id AND type = 2")
            # result_likes = self.db.execute(query_likes, {"user_id": user_id}).fetchall()
            # instructor_user_ids = {row[0] for row in result_likes}
            # # Need to map instructor user IDs back to instructor IDs if necessary
            return [row[0] for row in result] # Returning only based on enrollment for now
        except Exception as e:
            logger.error(f"Error fetching interacted instructors for user {user_id}: {e}", exc_info=True)
            return []

    def _get_instructor_details_by_ids(self, instructor_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch detailed information for a list of instructor IDs."""
        if not instructor_ids:
            return []
        try:
            placeholders = ",".join([f":id_{i}" for i in range(len(instructor_ids))])
            params = {f"id_{i}": inst_id for i, inst_id in enumerate(instructor_ids)}

            # Query matching the SQL fallback for consistency in returned fields
            query = text(f"""
                SELECT
                    i.id,
                    u.nickname as name,
                    u.bio,
                    u.profileUrl,
                    exp.name as specialization,
                    COALESCE(AVG(cr.rating), 0) as rating, -- Overall average rating
                    COUNT(DISTINCT c.id) as course_count,
                    COUNT(DISTINCT e.userId) as total_enrollments
                FROM instructor i
                JOIN `user` u ON i.userId = u.id
                LEFT JOIN expertise exp ON i.expertiseId = exp.id
                LEFT JOIN course c ON i.id = c.instructorId
                LEFT JOIN courseEnrollment e ON c.id = e.courseId
                LEFT JOIN courseRating cr ON c.id = cr.courseId
                WHERE i.id IN ({placeholders})
                GROUP BY i.id, u.nickname, u.bio, u.profileUrl, exp.name
            """)

            result = self.db.execute(query, params).fetchall()
            instructor_details = []
            for row in result:
                instructor_details.append({
                    "id": row.id,
                    "name": row.name,
                    "bio": row.bio,
                    "profile_url": row.profileUrl,
                    "specialization": row.specialization,
                    "rating": float(row.rating) if row.rating is not None else 0.0,
                    "course_count": row.course_count if row.course_count is not None else 0,
                    "total_enrollments": row.total_enrollments if row.total_enrollments is not None else 0
                })

            # Preserve order
            details_map = {detail['id']: detail for detail in instructor_details}
            ordered_details = [details_map[iid] for iid in instructor_ids if iid in details_map]
            return ordered_details

        except Exception as e:
            logger.error(f"Error fetching instructor details for IDs {instructor_ids}: {e}", exc_info=True)
            return []

    def _get_user_data(self, user_id: int):
        """
        사용자 데이터 조회 (백엔드 스키마 반영)
        """
        query = text("""
            SELECT 
                COUNT(DISTINCT ce.courseId) as completed_courses,
                AVG(cr.rating) as avg_rating_given,
                COUNT(DISTINCT c.categoryId) as preferred_categories,
                SUM(ce.progress) as study_time
            FROM `user` u
            LEFT JOIN courseEnrollment ce ON u.id = ce.userId
            LEFT JOIN courseRating cr ON u.id = cr.userId
            LEFT JOIN course c ON ce.courseId = c.id
            WHERE u.id = :user_id AND u.isDel = false
            GROUP BY u.id
        """)
        
        result = self.db.execute(query, {"user_id": user_id}).fetchone()
        return {
            'completed_courses': result[0] if result and result[0] is not None else 0,
            'avg_rating_given': float(result[1]) if result and result[1] is not None else 0.0,
            'preferred_categories': result[2] if result and result[2] is not None else 0,
            'study_time': float(result[3]) if result and result[3] is not None else 0.0
        }
        
    def _get_course_data(self):
        """
        강의 데이터 조회 (백엔드 스키마 반영)
        """
        query = text("""
            SELECT 
                c.id,
                c.price,
                COALESCE(AVG(cr.rating), 0) as rating,
                COUNT(DISTINCT ce.userId) as student_count,
                c.categoryId
            FROM course c
            LEFT JOIN courseRating cr ON c.id = cr.courseId
            LEFT JOIN courseEnrollment ce ON c.id = ce.courseId
            WHERE c.status = 'ACTIVE' -- Assuming ACTIVE status exists and is correct
            GROUP BY c.id, c.categoryId -- Include categoryId in GROUP BY
        """)
        
        results = self.db.execute(query).fetchall()
        return [{
            'id': row[0],
            'price': row[1] if row[1] is not None else 0,
            'rating': float(row[2]) if row[2] is not None else 0.0,
            'student_count': row[3] if row[3] is not None else 0,
            'category_id': row[4]
        } for row in results]
        
    def _get_instructor_data(self):
        """
        강사 데이터 조회 (백엔드 스키마 반영)
        """
        query = text("""
            SELECT 
                i.id,
                u.nickname as name, -- Get name from user table
                u.bio as bio,       -- Get bio from user table
                COALESCE(AVG(cr.rating), 0) as avg_rating,
                COUNT(DISTINCT ce.userId) as total_students,
                COUNT(DISTINCT c.id) as course_count,
                COUNT(DISTINCT l.id) as follower_count -- Count likes targeting the instructor's user ID
            FROM instructor i
            JOIN `user` u ON i.userId = u.id -- Join instructor with user
            LEFT JOIN course c ON i.id = c.instructorId
            LEFT JOIN courseRating cr ON c.id = cr.courseId
            LEFT JOIN courseEnrollment ce ON c.id = ce.courseId
            LEFT JOIN likeTable l ON u.id = l.targetUserId AND l.type = 2 -- Assuming type=2 means instructor like
            WHERE u.isDel = false -- Check if user (instructor) is deleted
            GROUP BY i.id, u.nickname, u.bio -- Group by selected non-aggregated columns
        """)
        
        results = self.db.execute(query).fetchall()
        return [{
            'id': row[0],
            'name': row[1], # Use index based on SELECT order
            'bio': row[2],
            'avg_rating': float(row[3]) if row[3] is not None else 0.0,
            'total_students': row[4] if row[4] is not None else 0,
            'course_count': row[5] if row[5] is not None else 0,
            'follower_count': row[6] if row[6] is not None else 0
        } for row in results]
        
    def _get_popular_course_recommendations_sql(self, user_id: int, n_recommendations=10) -> List[Dict[str, Any]]:
        """Fallback method using SQL query based on popularity (rating, enrollment)."""
        logger.info(f"Executing SQL-based popular course recommendation query for user {user_id}")
        try:
            query = text(f"""
                SELECT
                    c.id,
                    c.subject as title,
                    c.description,
                    c.price,
                    c.thumbnailUrl,
                    COALESCE(AVG(cr.rating), 0) as rating,
                    cat.name as category_name,
                    u.nickname as instructor_name,
                    COUNT(DISTINCT e.userId) as enrollment_count
                FROM course c
                JOIN instructor i ON c.instructorId = i.id
                JOIN `user` u ON i.userId = u.id
                JOIN category cat ON c.categoryId = cat.id
                LEFT JOIN courseEnrollment e ON c.id = e.courseId
                LEFT JOIN courseRating cr ON c.id = cr.courseId
                WHERE c.status = 'ACTIVE'
                  AND u.isDel = false
                  AND c.id NOT IN (
                      SELECT ce_sub.courseId
                      FROM courseEnrollment ce_sub
                      WHERE ce_sub.userId = :user_id
                  )
                GROUP BY c.id, c.subject, c.description, c.price, c.thumbnailUrl, cat.name, u.nickname
                ORDER BY
                    rating DESC,
                    enrollment_count DESC
                LIMIT :limit
            """)

            result = self.db.execute(query, {"user_id": user_id, "limit": n_recommendations}).fetchall()
            recommendations = []
            for row in result:
                recommendations.append({
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "price": float(row.price) if row.price is not None else 0.0,
                    "thumbnail_url": row.thumbnailUrl,
                    "rating": float(row.rating) if row.rating is not None else 0.0,
                    "category": row.category_name,
                    "instructor_name": row.instructor_name,
                    "enrollment_count": row.enrollment_count if row.enrollment_count is not None else 0
                })
            logger.info(f"Found {len(recommendations)} SQL-based fallback recommendations for user {user_id}")
            return recommendations
        except Exception as e:
            logger.error(f"Error in SQL-based fallback recommendations for user {user_id}: {str(e)}", exc_info=True)
            return []

    def _get_popular_instructor_recommendations_sql(self, user_id: int, n_recommendations=5) -> List[Dict[str, Any]]:
        """Fallback method using SQL query based on popularity (rating, enrollment) for instructors."""
        logger.info(f"Executing SQL-based popular instructor recommendation query for user {user_id}")
        try:
            # This is the previously existing SQL query from get_instructor_recommendations
            query = text(f"""
                SELECT
                    i.id,
                    u.nickname as name,
                    u.bio,
                    u.profileUrl,
                    exp.name as specialization,
                    COALESCE(AVG(cr.rating), 0) as rating,
                    COUNT(DISTINCT c.id) as course_count,
                    COUNT(DISTINCT e.userId) as total_enrollments
                FROM instructor i
                JOIN `user` u ON i.userId = u.id
                LEFT JOIN expertise exp ON i.expertiseId = exp.id
                LEFT JOIN course c ON i.id = c.instructorId
                LEFT JOIN courseEnrollment e ON c.id = e.courseId
                LEFT JOIN courseRating cr ON c.id = cr.courseId
                WHERE u.isDel = false
                  AND i.id NOT IN (
                      SELECT DISTINCT c_sub.instructorId
                      FROM courseEnrollment ce_sub
                      JOIN course c_sub ON ce_sub.courseId = c_sub.id
                      WHERE ce_sub.userId = :user_id
                  )
                GROUP BY i.id, u.nickname, u.bio, u.profileUrl, exp.name
                ORDER BY
                    rating DESC,
                    total_enrollments DESC
                LIMIT :limit
            """)
            result = self.db.execute(query, {"user_id": user_id, "limit": n_recommendations}).fetchall()
            recommendations = []
            for row in result:
                recommendations.append({
                    "id": row.id,
                    "name": row.name,
                    "bio": row.bio,
                    "profile_url": row.profileUrl,
                    "specialization": row.specialization,
                    "rating": float(row.rating) if row.rating is not None else 0.0,
                    "course_count": row.course_count if row.course_count is not None else 0,
                    "total_enrollments": row.total_enrollments if row.total_enrollments is not None else 0
                })
            logger.info(f"Found {len(recommendations)} SQL-based fallback instructor recommendations for user {user_id}")
            return recommendations
        except Exception as e:
            logger.error(f"Error in SQL-based fallback instructor recommendations for user {user_id}: {str(e)}", exc_info=True)
            return []

    def _get_all_courses(self) -> pd.DataFrame:
        """Fetch all active courses with their features."""
        try:
            query = text("""
                SELECT 
                    c.id,
                    c.subject as title,
                    c.description,
                    c.categoryId as category_id
                FROM course c
                JOIN instructor i ON c.instructorId = i.id
                JOIN `user` u ON i.userId = u.id
                WHERE c.status = 'ACTIVE' AND u.isDel = false
            """)
            result = self.db.execute(query).fetchall()
            
            if not result:
                logger.warning("No active courses found for content-based recommendations.")
                return pd.DataFrame(columns=['id', 'title', 'description', 'category_id'])
                
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=['id', 'title', 'description', 'category_id'])
            
            # Ensure correct data types
            df['id'] = df['id'].astype(int)
            df['title'] = df['title'].astype(str)
            df['description'] = df['description'].astype(str)
            df['category_id'] = df['category_id'].astype(int)
            
            logger.info(f"Loaded {len(df)} courses for content-based recommendations.")
            return df
        except Exception as e:
            logger.error(f"Error fetching course data: {e}", exc_info=True)
            return pd.DataFrame(columns=['id', 'title', 'description', 'category_id'])

    def get_course_recommendations(self, user_id: int, n_recommendations=10) -> List[Dict[str, Any]]:
        """
        사용자의 학습 이력과 선호도를 기반으로 강의 추천 (하이브리드 접근 방식)
        """
        recommendations = []
        ml_recommendations_generated = False
        content_based_recommendations_generated = False

        try:
            # 0. Check if the collaborative filtering model is trained
            if self.course_recommender.user_item_matrix is None:
                logger.warning("Collaborative filtering model not trained or training failed.")
                # Proceed to content-based filtering
            else:
                # 1. Check if user exists
                user_check_query = text("SELECT id FROM `user` WHERE id = :user_id AND isDel = false")
                user = self.db.execute(user_check_query, {"user_id": user_id}).fetchone()
                if not user:
                    logger.warning(f"User with ID {user_id} not found or deleted")
                    return [] # Return empty if user doesn't exist

                # 2. Get all active course IDs
                all_active_ids = self._get_all_active_course_ids()
                if not all_active_ids:
                    logger.warning("No active courses found.")
                    # Proceed to content-based filtering
                else:
                    # 3. Get course IDs user is already enrolled in
                    enrolled_ids = self._get_user_enrolled_course_ids(user_id)
                    enrolled_ids_set = set(enrolled_ids)

                    # 4. Determine candidate course IDs (active courses not enrolled by the user)
                    candidate_ids = [cid for cid in all_active_ids if cid not in enrolled_ids_set]

                    if not candidate_ids:
                        logger.info(f"User {user_id} has no candidate courses for collaborative filtering.")
                        # Proceed to content-based filtering
                    else:
                        # 5. Predict ratings for candidate courses using the collaborative filtering model
                        logger.info(f"Predicting ratings for user {user_id} using collaborative filtering for {len(candidate_ids)} candidates.")
                        predictions = self.course_recommender.predict(user_id, candidate_ids)

                        # 6. Get top N recommendations based on predicted rating
                        top_n_predictions = predictions[:n_recommendations]
                        top_n_course_ids = [item_id for item_id, est_rating in top_n_predictions]

                        if not top_n_course_ids:
                            logger.info(f"Collaborative filtering generated no recommendations for user {user_id}. Proceeding to content-based filtering.")
                            # Proceed to content-based filtering
                        else:
                            # 7. Fetch details for the recommended courses
                            logger.info(f"Fetching details for top {len(top_n_course_ids)} collaborative filtering recommended course IDs: {top_n_course_ids}")
                            recommendations = self._get_course_details_by_ids(top_n_course_ids)
                            ml_recommendations_generated = True # Mark that collaborative filtering recommendations were successful
                            logger.info(f"Found {len(recommendations)} collaborative filtering based course recommendations for user {user_id}")

            # --- Content-Based Filtering ---
            if not ml_recommendations_generated and self.content_based_recommender.similarity_matrix is not None:
                logger.info(f"Attempting content-based filtering for user {user_id}.")
                
                # Get user's enrolled courses
                enrolled_ids = self._get_user_enrolled_course_ids(user_id)
                
                if enrolled_ids:
                    # Get content-based recommendations based on user's enrolled courses
                    content_based_predictions = self.content_based_recommender.recommend_for_user(enrolled_ids, n_recommendations)
                    
                    if content_based_predictions:
                        content_based_course_ids = [course_id for course_id, score in content_based_predictions]
                        logger.info(f"Found {len(content_based_course_ids)} content-based recommendations for user {user_id}.")
                        
                        # Fetch details for the recommended courses
                        content_based_recommendations = self._get_course_details_by_ids(content_based_course_ids)
                        
                        if content_based_recommendations:
                            recommendations = content_based_recommendations
                            content_based_recommendations_generated = True
                            logger.info(f"Successfully generated {len(recommendations)} content-based recommendations for user {user_id}.")
                        else:
                            logger.warning("Content-based recommendations found but failed to fetch course details.")
                    else:
                        logger.warning("Content-based filtering generated no recommendations.")
                else:
                    logger.warning(f"User {user_id} has no enrolled courses for content-based filtering.")

            # --- Fallback Logic --- 
            if not ml_recommendations_generated and not content_based_recommendations_generated:
                logger.warning(f"Both collaborative filtering and content-based filtering failed for user {user_id}. Using SQL-based fallback.")
                recommendations = self._get_popular_course_recommendations_sql(user_id, n_recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Unexpected error in get_course_recommendations for user {user_id}: {str(e)}", exc_info=True)
            # Final safety net: attempt fallback on any unexpected error
            try:
                if not ml_recommendations_generated and not content_based_recommendations_generated:
                     logger.error(f"Attempting SQL fallback due to unexpected error for user {user_id}.")
                     return self._get_popular_course_recommendations_sql(user_id, n_recommendations)
                else:
                    return [] # Return empty if ML part succeeded but error occurred later
            except Exception as fallback_e:
                 logger.error(f"Error during fallback attempt for user {user_id}: {str(fallback_e)}", exc_info=True)
                 return [] # Return empty if fallback also fails

    def get_instructor_recommendations(self, user_id: int, n_recommendations=5) -> List[Dict[str, Any]]:
        """
        사용자의 학습 이력과 선호도를 기반으로 강사 추천 (머신러닝 모델 우선, 실패 시 인기 기반 대체)
        """
        recommendations = []
        ml_recommendations_generated = False

        try:
            # 0. Check if the instructor model is trained
            if self.instructor_recommender.user_instructor_matrix is None:
                logger.warning("Instructor Recommender model not trained or training failed.")
                # Proceed to fallback directly
            else:
                # 1. Check if user exists (redundant if called after course rec, but good practice)
                user_check_query = text("SELECT id FROM `user` WHERE id = :user_id AND isDel = false")
                user = self.db.execute(user_check_query, {"user_id": user_id}).fetchone()
                if not user:
                    logger.warning(f"User {user_id} not found for instructor recommendations.")
                    return []

                # 2. Get all active instructor IDs
                all_active_instructor_ids = self._get_all_active_instructor_ids()
                if not all_active_instructor_ids:
                    logger.warning("No active instructors found.")
                    # Proceed to fallback
                else:
                    # 3. Get instructors user has already interacted with (e.g., enrolled in their courses)
                    interacted_instructor_ids = self._get_instructors_user_interacted_with(user_id)
                    interacted_ids_set = set(interacted_instructor_ids)

                    # 4. Determine candidate instructor IDs
                    candidate_ids = [iid for iid in all_active_instructor_ids if iid not in interacted_ids_set]

                    if not candidate_ids:
                        logger.info(f"User {user_id} has no candidate instructors for ML recommendation.")
                        # Proceed to fallback
                    else:
                        # 5. Predict preference scores for candidate instructors
                        logger.info(f"Predicting instructor scores for user {user_id} using NMF model for {len(candidate_ids)} candidates.")
                        predictions = self.instructor_recommender.predict(user_id, candidate_ids)

                        # 6. Get top N recommendations
                        top_n_predictions = predictions[:n_recommendations]
                        top_n_instructor_ids = [item_id for item_id, score in top_n_predictions]

                        if not top_n_instructor_ids:
                            logger.info(f"NMF model generated no instructor recommendations for user {user_id}. Proceeding to fallback.")
                            # Proceed to fallback
                        else:
                            # 7. Fetch details for recommended instructors
                            logger.info(f"Fetching details for top {len(top_n_instructor_ids)} NMF-recommended instructor IDs: {top_n_instructor_ids}")
                            recommendations = self._get_instructor_details_by_ids(top_n_instructor_ids)
                            ml_recommendations_generated = True
                            logger.info(f"Found {len(recommendations)} ML-based instructor recommendations for user {user_id}")

            # --- Fallback Logic --- 
            if not ml_recommendations_generated:
                logger.warning(f"ML instructor recommendation failed or produced no results for user {user_id}. Using SQL-based fallback.")
                recommendations = self._get_popular_instructor_recommendations_sql(user_id, n_recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Unexpected error in get_instructor_recommendations for user {user_id}: {str(e)}", exc_info=True)
            # Final safety net: attempt fallback
            try:
                if not ml_recommendations_generated:
                     logger.error(f"Attempting SQL fallback for instructors due to unexpected error for user {user_id}.")
                     return self._get_popular_instructor_recommendations_sql(user_id, n_recommendations)
                else:
                    return []
            except Exception as fallback_e:
                 logger.error(f"Error during instructor fallback attempt for user {user_id}: {str(fallback_e)}", exc_info=True)
                 return [] 