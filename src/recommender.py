from pathlib import Path
import re
import joblib as jb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances


def clean_movies_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza os dados brutos de filmes."""
    df_clean = df.copy()

    # Padroniza nomes de colunas
    df_clean.columns = [c.upper() for c in df_clean.columns]

    # Remove duplicados por título
    df_clean = df_clean.drop_duplicates(subset=["TITLE"])

    # Converte textos para maiúsculo
    df_clean["TITLE"] = df_clean["TITLE"].astype(str).str.upper()
    df_clean["GENRES"] = df_clean["GENRES"].astype(str).str.upper()

    # Extrai o ano do título (ex: "FILME (1995)" -> 1995)
    extracted_year = df_clean["TITLE"].str.extract(r"\((\d{4})\)\s*$")[0]
    df_clean["ANO_FILME"] = pd.to_numeric(extracted_year, errors="coerce")

    # Filtra filmes sem gênero válido ou sem ano
    valid_mask = (
        (df_clean["GENRES"] != "(NO GENRES LISTED)")
        & (df_clean["ANO_FILME"].notna())
    )
    df_clean = df_clean[valid_mask].copy()

    df_clean["ANO_FILME"] = df_clean["ANO_FILME"].astype(int)
    return df_clean


def train_recommender_model(
    df_clean: pd.DataFrame, n_clusters: int = 3, random_state: int = 123
) -> dict:
    """Treina o modelo KMeans nos gêneros e rotula a base."""
    # One-hot encoding dos gêneros
    df_generos_ohe = df_clean["GENRES"].str.get_dummies("|")
    colunas_modelo = list(df_generos_ohe.columns)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(df_generos_ohe)

    df_base = df_clean.copy()
    df_base["CLUSTER"] = kmeans.predict(df_generos_ohe)

    return {
        "BASE_DADOS": df_base,
        "VARIAVEIS": colunas_modelo,
        "MODELO": kmeans,
    }


def get_recommendations(
    df_movies: pd.DataFrame,
    model_variables: list,
    movie_title: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Gera recomendações usando cálculo vetorizado de distância euclidiana."""
    filme_selecionado = df_movies[df_movies["TITLE"] == movie_title]

    if filme_selecionado.empty:
        return pd.DataFrame(columns=list(df_movies.columns) + ["SIMILARIDADE"])

    cluster_filme = filme_selecionado["CLUSTER"].values[0]

    # Filtra filmes do mesmo cluster, excluindo o filme de origem
    df_rec = df_movies[
        (df_movies["CLUSTER"] == cluster_filme)
        & (df_movies["TITLE"] != movie_title)
    ].copy()

    if df_rec.empty:
        df_rec["SIMILARIDADE"] = []
        return df_rec

    # Matriz OHE do filme de referência
    genero_filme_selec = (
        filme_selecionado["GENRES"]
        .str.get_dummies(sep="|")
        .reindex(columns=model_variables, fill_value=0)
    )

    # Matriz OHE de todos os candidatos de uma vez (vetorizado)
    genero_filmes_candidatos = (
        df_rec["GENRES"]
        .str.get_dummies(sep="|")
        .reindex(columns=model_variables, fill_value=0)
    )

    # Distância Euclidiana em bloco (operações em C/BLAS)
    distancias = euclidean_distances(genero_filme_selec, genero_filmes_candidatos)[0]

    df_rec["SIMILARIDADE"] = distancias
    return df_rec.sort_values("SIMILARIDADE").head(top_n)


def filter_movies(
    df: pd.DataFrame,
    search_query: str = "",
    genres: list = None,
    year_range: tuple = None,
    cluster = None,
) -> pd.DataFrame:
    """Filtra a base de filmes por busca textual, gêneros, anos e cluster."""
    filtered = df.copy()

    if search_query:
        query_upper = str(search_query).strip().upper()
        filtered = filtered[filtered["TITLE"].str.contains(query_upper, case=False, na=False)]

    if genres:
        for g in genres:
            g_upper = str(g).strip().upper()
            filtered = filtered[filtered["GENRES"].str.contains(g_upper, case=False, na=False)]

    if year_range:
        min_year, max_year = year_range
        filtered = filtered[
            (filtered["ANO_FILME"] >= min_year) & (filtered["ANO_FILME"] <= max_year)
        ]

    if cluster is not None and str(cluster).lower() != "todos":
        filtered = filtered[filtered["CLUSTER"] == int(cluster)]

    return filtered


def train_and_save_pipeline(
    csv_path: str, output_path: str = "modelo.pkl", n_clusters: int = 3
) -> dict:
    """Executa o pipeline completo de carregamento, limpeza, treino e salvamento."""
    print(f"Lendo base de dados em: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    print("Limpando e formatando os dados...")
    df_clean = clean_movies_data(df_raw)

    print(f"Treinando modelo KMeans (k={n_clusters})...")
    model_bundle = train_recommender_model(df_clean, n_clusters=n_clusters)

    print(f"Salvando artefato em: {output_path}")
    jb.dump(model_bundle, output_path)

    return model_bundle


if __name__ == "__main__":
    csv_file = Path("movie.csv")
    if csv_file.exists():
        train_and_save_pipeline(str(csv_file), "modelo.pkl")
    else:
        print(f"Arquivo não encontrado: {csv_file}")
