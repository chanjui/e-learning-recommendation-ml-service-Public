import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import logging

logger = logging.getLogger(__name__)

class InstructorRecommender:
    def __init__(self, n_components=10, max_iter=500, random_state=42):
        """Initialize the recommender with NMF model for instructors."""
        # Reduced components slightly for potentially sparser instructor data
        self.model = NMF(n_components=n_components, init='random', max_iter=max_iter, random_state=random_state, l1_ratio=0)
        self.user_instructor_matrix = None
        self.user_map = None
        self.instructor_map = None
        self.user_factors = None
        self.instructor_factors = None
        self.user_map_reverse = None
        self.instructor_map_reverse = None

    def fit(self, avg_ratings_df: pd.DataFrame):
        """Create user-instructor matrix from average ratings and fit the NMF model."""
        if avg_ratings_df.empty:
            logger.warning("Received empty avg ratings dataframe for instructor training. Model will not be trained.")
            return

        # Expecting columns: userId, instructorId, avg_rating
        if not all(col in avg_ratings_df.columns for col in ['userId', 'instructorId', 'avg_rating']):
            raise ValueError("Average Ratings DataFrame must contain 'userId', 'instructorId', and 'avg_rating' columns.")

        logger.info(f"Creating user-instructor matrix from {len(avg_ratings_df)} average ratings.")
        try:
            # Create user-instructor matrix (pivot table)
            # Fill missing values with 0 (or global average if preferred, but 0 is simpler for NMF)
            self.user_instructor_matrix = avg_ratings_df.pivot_table(index='userId', columns='instructorId', values='avg_rating').fillna(0)

            # Create mappings
            self.user_map = {user_id: i for i, user_id in enumerate(self.user_instructor_matrix.index)}
            self.instructor_map = {inst_id: i for i, inst_id in enumerate(self.user_instructor_matrix.columns)}
            self.user_map_reverse = {i: user_id for user_id, i in self.user_map.items()}
            self.instructor_map_reverse = {i: inst_id for inst_id, i in self.instructor_map.items()}

            logger.info("Training NMF model for instructors...")
            # Fit the model
            self.user_factors = self.model.fit_transform(self.user_instructor_matrix)
            self.instructor_factors = self.model.components_.T # Transpose
            logger.info("NMF instructor model training complete.")

        except Exception as e:
            logger.error(f"Error during NMF instructor model training: {str(e)}", exc_info=True)
            self.user_instructor_matrix = None # Indicate failure

    def predict(self, user_id: int, instructor_ids: list[int]) -> list[tuple[int, float]]:
        """Predict preference scores for a list of instructors for a given user."""
        if self.user_instructor_matrix is None or self.user_factors is None or self.instructor_factors is None:
            logger.warning("NMF Instructor Model not trained or training failed. Cannot make predictions.")
            return []

        if user_id not in self.user_map:
            logger.warning(f"User ID {user_id} not found in instructor training data (user map). Cannot make personalized predictions.")
            return []

        user_index = self.user_map[user_id]
        user_vector = self.user_factors[user_index, :]

        predictions = []
        logger.debug(f"Predicting instructor scores for user {user_id} (index {user_index}) for candidate instructors.")

        for instructor_id in instructor_ids:
            if instructor_id not in self.instructor_map:
                continue # Skip instructors not seen during training

            instructor_index = self.instructor_map[instructor_id]
            instructor_vector = self.instructor_factors[instructor_index, :]

            # Predict score (dot product)
            predicted_score = np.dot(user_vector, instructor_vector)
            predictions.append((instructor_id, predicted_score))

        # Sort by predicted score
        predictions.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Generated {len(predictions)} NMF-based instructor predictions for user {user_id}")
        return predictions 