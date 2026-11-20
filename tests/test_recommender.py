import pytest
import pandas as pd
import numpy as np
from src.recommender import (
    clean_movies_data,
    train_recommender_model,
    get_recommendations,
    filter_movies,
)

@pytest.fixture
def sample_movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Movie No Year",
            "Unknown (2020)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama|Romance",
            "Comedy",
            "(no genres listed)"
        ]
    })

def test_clean_movies_data(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)

    assert len(cleaned) == 4
    assert "ANO_FILME" in cleaned.columns
    assert cleaned["ANO_FILME"].dtype == int
    assert list(cleaned["ANO_FILME"]) == [1995, 1995, 1995, 1995]
    assert "TITLE" in cleaned.columns
    assert cleaned["TITLE"].iloc[0] == "TOY STORY (1995)"

def test_train_recommender_model(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)

    assert "BASE_DADOS" in model_bundle
    assert "VARIAVEIS" in model_bundle
    assert "MODELO" in model_bundle

    assert "CLUSTER" in model_bundle["BASE_DADOS"].columns
    assert len(model_bundle["VARIAVEIS"]) > 0

def test_get_recommendations(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)

    df_base = model_bundle["BASE_DADOS"]
    variaveis = model_bundle["VARIAVEIS"]

    recs = get_recommendations(
        df_movies=df_base,
        model_variables=variaveis,
        movie_title="TOY STORY (1995)",
        top_n=2
    )

    assert len(recs) <= 2
    assert "SIMILARIDADE" in recs.columns
    assert "TOY STORY (1995)" not in recs["TITLE"].values
    assert recs["SIMILARIDADE"].is_monotonic_increasing

def test_filter_movies(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)
    df_base = model_bundle["BASE_DADOS"]

    # 1. Filtro por texto
    res_text = filter_movies(df_base, search_query="TOY")
    assert len(res_text) == 1
    assert res_text.iloc[0]["TITLE"] == "TOY STORY (1995)"

    # 2. Filtro por gênero
    res_genre = filter_movies(df_base, genres=["Romance"])
    assert len(res_genre) == 2
    assert all("ROMANCE" in g for g in res_genre["GENRES"])

    # 3. Filtro por intervalo de anos
    res_year = filter_movies(df_base, year_range=(1990, 1994))
    assert len(res_year) == 0

    # 4. Filtro por cluster
    cluster_val = df_base.iloc[0]["CLUSTER"]
    res_cluster = filter_movies(df_base, cluster=cluster_val)
    assert all(c == cluster_val for c in res_cluster["CLUSTER"])
