
from recommender.model import NetflixRecommender


def evaluate_title(recommender, title, top_n=10):
    data = recommender.data

    source = data[
        data["title"].str.casefold() == title.casefold()
    ]

    if source.empty:
        print(f"\nSkipping '{title}': title not found.")
        return

    source = source.iloc[0]
    results = recommender.recommend(title, top_n=top_n)

    if results.empty:
        print(f"\nNo recommendations for '{title}'.")
        return

    source_categories = {
        item.strip().casefold()
        for item in str(source.get("listed_in", "")).split(",")
        if item.strip()
    }

    overlaps = []

    for _, result in results.iterrows():
        recommended_categories = {
            item.strip().casefold()
            for item in str(result.get("listed_in", "")).split(",")
            if item.strip()
        }

        overlaps.append(
            bool(source_categories & recommended_categories)
        )

    average_similarity = results["similarity_score"].mean()
    overlap_rate = sum(overlaps) / len(overlaps) * 100

    print(f"\nTitle: {title}")
    print(f"Recommendations evaluated: {len(results)}")
    print(f"Average similarity score: {average_similarity:.4f}")
    print(f"Category overlap: {overlap_rate:.1f}%")
    print("\nRecommended titles:")
    print(
        results[
            ["title", "listed_in", "similarity_score"]
        ].to_string(index=False)
    )


def main():
    recommender = NetflixRecommender()

    test_titles = [
        "Ganglands",
        "Dick Johnson Is Dead",
        "Lupin",
    ]

    print("Netflix Recommendation Evaluation")
    print("=" * 36)

    for title in test_titles:
        evaluate_title(recommender, title)


if __name__ == "__main__":
    main()
