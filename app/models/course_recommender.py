import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import logging

logger = logging.getLogger(__name__)

class CourseRecommender:
    def __init__(self, n_components=15, max_iter=500, random_state=42):
        """Initialize the recommender with NMF model."""
        # n_components: number of latent factors
        self.model = NMF(n_components=n_components, init='random', max_iter=max_iter, random_state=random_state, l1_ratio=0)
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None
        self.user_factors = None
        self.item_factors = None
        self.user_map_reverse = None
        self.item_map_reverse = None

    def fit(self, ratings_df: pd.DataFrame):
        """Create user-item matrix and fit the NMF model."""
        if ratings_df.empty:
            logger.warning("Received empty ratings dataframe for training. Model will not be trained.")
            return

        if not all(col in ratings_df.columns for col in ['userId', 'courseId', 'rating']):
            raise ValueError("Ratings DataFrame must contain 'userId', 'courseId', and 'rating' columns.")

        logger.info(f"Creating user-item matrix from {len(ratings_df)} ratings.")
        try:
            # Create user-item matrix (pivot table)
            # Fill missing values with 0, as NMF requires non-negative inputs
            self.user_item_matrix = ratings_df.pivot_table(index='userId', columns='courseId', values='rating').fillna(0)

            # Create mappings for user and item IDs to matrix indices
            self.user_map = {user_id: i for i, user_id in enumerate(self.user_item_matrix.index)}
            self.item_map = {item_id: i for i, item_id in enumerate(self.user_item_matrix.columns)}
            self.user_map_reverse = {i: user_id for user_id, i in self.user_map.items()}
            self.item_map_reverse = {i: item_id for item_id, i in self.item_map.items()}

            logger.info("Training NMF model...")
            # Fit the model
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.item_factors = self.model.components_.T # Transpose components to get item factors
            logger.info("NMF model training complete.")

        except Exception as e:
            logger.error(f"Error during NMF model training: {str(e)}", exc_info=True)
            self.user_item_matrix = None # Indicate failure

    def predict(self, user_id: int, item_ids: list[int]) -> list[tuple[int, float]]:
        """Predict ratings for a list of items for a given user using the trained NMF model."""
        if self.user_item_matrix is None or self.user_factors is None or self.item_factors is None:
            logger.warning("NMF Model not trained or training failed. Cannot make predictions.")
            return []

        if user_id not in self.user_map:
            logger.warning(f"User ID {user_id} not found in training data (user map). Cannot make personalized predictions.")
            return []

        user_index = self.user_map[user_id]
        user_vector = self.user_factors[user_index, :]

        predictions = []
        logger.debug(f"Predicting ratings for user {user_id} (index {user_index}) for candidate items.")

        for item_id in item_ids:
            if item_id not in self.item_map:
                # logger.debug(f"Item ID {item_id} not found in training data (item map). Skipping.")
                continue # Skip items not seen during training

            item_index = self.item_map[item_id]
            item_vector = self.item_factors[item_index, :]

            # Predict rating by taking the dot product of user and item factor vectors
            predicted_rating = np.dot(user_vector, item_vector)
            predictions.append((item_id, predicted_rating))

        # Sort predictions by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Generated {len(predictions)} NMF-based predictions for user {user_id}")
        return predictions 