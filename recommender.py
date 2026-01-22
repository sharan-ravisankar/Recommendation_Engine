import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv("movies.csv")
        self.ratings = pd.read_csv("ratings.csv")

        self._build_content_model()
        self._build_item_model()
        self._build_mappings()

    def _build_content_model(self):
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ", regex=False)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["genres"])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        self.content_similarity_df = pd.DataFrame(
            cosine_sim,
            index=self.movies.movieId,
            columns=self.movies.movieId
        )

    def _build_item_model(self):
        item_user_matrix = self.ratings.pivot_table(
            index="movieId",
            columns="userId",
            values="rating"
        ).fillna(0)

        item_similarity = cosine_similarity(item_user_matrix)

        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )

    def _build_mappings(self):
        self.title_to_id = pd.Series(
            self.movies.movieId.values, index=self.movies.title
        )
        self.id_to_title = pd.Series(
            self.movies.title.values, index=self.movies.movieId
        )

    def recommend_content(self, movie_title, top_n=10):
        movie_id = self.title_to_id[movie_title]
        scores = self.content_similarity_df[movie_id].sort_values(ascending=False)
        scores = scores.drop(movie_id)
        return self.id_to_title.loc[scores.head(top_n).index].tolist()

    def recommend_item(self, movie_title, top_n=10):
        movie_id = self.title_to_id[movie_title]
        scores = self.item_similarity_df[movie_id].sort_values(ascending=False)
        scores = scores.drop(movie_id)
        return self.id_to_title.loc[scores.head(top_n).index].tolist()
