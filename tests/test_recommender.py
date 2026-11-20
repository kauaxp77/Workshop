import pytest
import pandas as pd
import numpy as np
from src.recommender import (
    clean_movies_data,
    train_recommender_model,
    get_recommendations,
    filter_movies,
    evaluate_genre_precision,
)

@pytest.fixture
def sample_movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6, 7],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Toy Story 2 (1999)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Movie No Year",
            "Unknown (2020)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama|Romance",
            "Comedy",
            "(no genres listed)"
        ]
    })

def test_clean_movies_data(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)

    assert len(cleaned) == 5
    assert "ANO_FILME" in cleaned.columns
    assert cleaned["ANO_FILME"].dtype == int
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

def test_get_recommendations_cosine_and_tiebreak(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)

    df_base = model_bundle["BASE_DADOS"]
    variaveis = model_bundle["VARIAVEIS"]

    recs = get_recommendations(
        df_movies=df_base,
        model_variables=variaveis,
        movie_title="TOY STORY (1995)",
        top_n=3,
        metric="cosine",
        same_cluster_only=False,
    )

    assert len(recs) <= 3
    assert "SIMILARIDADE" in recs.columns
    assert "TOY STORY (1995)" not in recs["TITLE"].values

    # Na similaridade de cosseno, maior valor = mais parecido (monotonicamente decrescente)
    assert recs["SIMILARIDADE"].is_monotonic_decreasing
    # Valores normalizados entre 0 e 1
    assert all(0.0 <= s <= 1.0001 for s in recs["SIMILARIDADE"])

    # Toy Story 2 tem exatamente os mesmos gêneros de Toy Story 1 (similaridade 1.0)
    top_movie = recs.iloc[0]
    assert top_movie["TITLE"] == "TOY STORY 2 (1999)"
    assert np.isclose(top_movie["SIMILARIDADE"], 1.0)

def test_get_recommendations_cluster_filter_option(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)
    df_base = model_bundle["BASE_DADOS"]
    variaveis = model_bundle["VARIAVEIS"]

    # Com same_cluster_only=True, todos os recomendados devem ter o mesmo cluster do filme
    recs_same = get_recommendations(
        df_movies=df_base,
        model_variables=variaveis,
        movie_title="TOY STORY (1995)",
        top_n=5,
        same_cluster_only=True,
    )
    cluster_toy = df_base.loc[df_base["TITLE"] == "TOY STORY (1995)", "CLUSTER"].values[0]
    assert all(c == cluster_toy for c in recs_same["CLUSTER"])

def test_evaluate_genre_precision():
    target = "ADVENTURE|ANIMATION|COMEDY"
    recs_genres = pd.Series([
        "ADVENTURE|ANIMATION|COMEDY", # 3/3 = 1.0
        "ADVENTURE|COMEDY",           # 2/2 = 1.0 (ou 2/3 de recall)
        "DRAMA|ROMANCE",              # 0/2 = 0.0
    ])

    precision = evaluate_genre_precision(target, recs_genres)
    assert 0.0 <= precision <= 1.0
    assert precision > 0.5

def test_filter_movies(sample_movies_df):
    cleaned = clean_movies_data(sample_movies_df)
    model_bundle = train_recommender_model(cleaned, n_clusters=2, random_state=42)
    df_base = model_bundle["BASE_DADOS"]

    # 1. Filtro por texto
    res_text = filter_movies(df_base, search_query="TOY")
    assert len(res_text) == 2

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
