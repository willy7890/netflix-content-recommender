
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender.data import load_data
from recommender.features import prepare_features


class NetflixRecommender:
    """Content-based Netflix recommendation engine."""

    def __init__(self, dataset_path=None):
        if dataset_path is None:
            raw_data = load_data()
        else:
            raw_data = load_data(dataset_path)

        self.data = prepare_features(raw_data)

        # Give genre/category terms greater influence than other features.
        self.data["content_features"] = (
            self.data["listed_in"].fillna("").astype(str).str.lower()
            + " "
            + self.data["listed_in"].fillna("").astype(str).str.lower()
            + " "
            + self.data["director"].fillna("").astype(str).str.lower()
            + " "
            + self.data["country"].fillna("").astype(str).str.lower()
            + " "
            + self.data["type"].fillna("").astype(str).str.lower()
            + " "
            + self.data["rating"].fillna("").astype(str).str.lower()
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )

        self.feature_matrix = self.vectorizer.fit_transform(
            self.data["content_features"]
        )

        self.similarity_matrix = cosine_similarity(
            self.feature_matrix,
            dense_output=False,
        )

        self.title_to_indices = {}

        for index, title in enumerate(self.data["title"]):
            key = str(title).casefold()
            self.title_to_indices.setdefault(key, []).append(index)

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Recommend similar movies or TV shows."""

        if not isinstance(title, str) or not title.strip():
            raise ValueError("Please provide a valid title.")

        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        key = title.strip().casefold()

        if key not in self.title_to_indices:
            matches = self.data[
                self.data["title"].str.contains(
                    title.strip(),
                    case=False,
                    regex=False,
                    na=False,
                )
            ]

            if matches.empty:
                raise ValueError(
                    f"Title '{title}' was not found in the dataset."
                )

            return matches[
                ["title", "type", "release_year", "rating", "listed_in"]
            ].head(top_n).reset_index(drop=True)

        # Use the first record if multiple entries share the same title.
        index = self.title_to_indices[key][0]
        selected = self.data.iloc[index]

        scores = self.similarity_matrix[index].toarray().ravel()

        # Prefer candidates that share at least one category.
        selected_categories = {
            category.strip()
            for category in str(selected.get("listed_in", "")).casefold().split(",")
            if category.strip()
        }

        recommendations = []

        for candidate_index, score in enumerate(scores):
            if candidate_index == index:
                continue

            candidate = self.data.iloc[candidate_index]

            if str(candidate["title"]).casefold() == key:
                continue

            candidate_categories = {
                category.strip()
                for category in str(candidate.get("listed_in", "")).casefold().split(",")
                if category.strip()
            }

            shared_categories = selected_categories & candidate_categories

            # Category overlap is a ranking signal, not a model accuracy score.
            category_bonus = (
                len(shared_categories) / len(selected_categories)
                if selected_categories
                else 0.0
            )

            adjusted_score = float(score) + (0.15 * category_bonus)

            recommendations.append(
                {
                    "title": candidate["title"],
                    "type": candidate.get("type", ""),
                    "release_year": candidate.get("release_year", None),
                    "rating": candidate.get("rating", ""),
                    "listed_in": candidate.get("listed_in", ""),
                    "similarity_score": round(float(score), 4),
                    "ranking_score": round(adjusted_score, 4),
                }
            )

        recommendations.sort(
            key=lambda item: item["ranking_score"],
            reverse=True,
        )

        results = pd.DataFrame(recommendations).head(top_n)

        return results.drop(columns=["ranking_score"]).reset_index(drop=True)


if __name__ == "__main__":
    recommender = NetflixRecommender()

    print("Netflix Content Recommender")
    print("---------------------------")

    title = input("Enter a movie or series title: ")

    try:
        results = recommender.recommend(title, top_n=10)

        if results.empty:
            print("No recommendations found.")
        else:
            print("\nRecommended titles:\n")
            print(results.to_string(index=False))

    except ValueError as error:
        print(error)
