import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self):
        """Initialize the content-based recommender."""
        self.course_features = None
        self.course_ids = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def fit(self, courses_df: pd.DataFrame):
        """Fit the content-based recommender with course features."""
        if courses_df.empty:
            logger.warning("Received empty courses dataframe for training. Model will not be trained.")
            return
            
        if not all(col in courses_df.columns for col in ['id', 'title', 'description', 'category_id']):
            raise ValueError("Courses DataFrame must contain 'id', 'title', 'description', and 'category_id' columns.")
            
        logger.info(f"Creating content-based features from {len(courses_df)} courses.")
        try:
            # Store course IDs for later reference
            self.course_ids = courses_df['id'].tolist()
            
            # Create a combined text feature from title, description, and category
            # This helps capture the semantic content of the course
            courses_df['combined_features'] = courses_df['title'] + ' ' + courses_df['description'] + ' ' + courses_df['category_id'].astype(str)
            
            # Create TF-IDF features
            self.course_features = self.vectorizer.fit_transform(courses_df['combined_features'])
            
            # Calculate cosine similarity between all courses
            self.similarity_matrix = cosine_similarity(self.course_features)
            
            logger.info("Content-based recommender training complete.")
            
        except Exception as e:
            logger.error(f"Error during content-based recommender training: {str(e)}", exc_info=True)
            self.course_features = None
            
    def recommend(self, course_id: int, n_recommendations: int = 10) -> list[tuple[int, float]]:
        """Recommend similar courses based on content similarity."""
        if self.similarity_matrix is None or self.course_ids is None:
            logger.warning("Content-based recommender not trained or training failed. Cannot make recommendations.")
            return []
            
        if course_id not in self.course_ids:
            logger.warning(f"Course ID {course_id} not found in training data. Cannot make content-based recommendations.")
            return []
            
        # Find the index of the course
        course_index = self.course_ids.index(course_id)
        
        # Get similarity scores for this course
        similarity_scores = self.similarity_matrix[course_index]
        
        # Create a list of (course_id, similarity_score) tuples
        recommendations = [(self.course_ids[i], similarity_scores[i]) for i in range(len(self.course_ids))]
        
        # Sort by similarity score (descending) and exclude the input course
        recommendations = [rec for rec in recommendations if rec[0] != course_id]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return recommendations[:n_recommendations]
        
    def recommend_for_user(self, user_courses: list[int], n_recommendations: int = 10) -> list[tuple[int, float]]:
        """Recommend courses based on a user's enrolled courses."""
        if self.similarity_matrix is None or self.course_ids is None:
            logger.warning("Content-based recommender not trained or training failed. Cannot make recommendations.")
            return []
            
        if not user_courses:
            logger.warning("No user courses provided. Cannot make content-based recommendations.")
            return []
            
        # Get recommendations for each user course
        all_recommendations = []
        for course_id in user_courses:
            if course_id in self.course_ids:
                course_recommendations = self.recommend(course_id, n_recommendations)
                all_recommendations.extend(course_recommendations)
                
        if not all_recommendations:
            logger.warning("No content-based recommendations found for user courses.")
            return []
            
        # Aggregate recommendations by course ID
        aggregated_recommendations = {}
        for course_id, score in all_recommendations:
            if course_id in aggregated_recommendations:
                aggregated_recommendations[course_id] += score
            else:
                aggregated_recommendations[course_id] = score
                
        # Convert to list and sort by score
        final_recommendations = [(course_id, score) for course_id, score in aggregated_recommendations.items()]
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return final_recommendations[:n_recommendations] 