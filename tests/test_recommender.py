
import pytest

from recommender.model import NetflixRecommender


@pytest.fixture(scope="module")
def recommender():
    return NetflixRecommender()


def test_ganglands_returns_recommendations(recommender):
    results = recommender.recommend("Ganglands", top_n=10)

    assert not results.empty
    assert len(results) <= 10
    assert "similarity_score" in results.columns


def test_selected_title_is_not_recommended_again(recommender):
    results = recommender.recommend("Ganglands", top_n=10)

    assert not results["title"].str.casefold().eq("ganglands").any()


def test_invalid_title_raises_error(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("This Title Does Not Exist 12345")


def test_invalid_top_n_raises_error(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("Ganglands", top_n=0)
