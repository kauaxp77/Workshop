# Sistema de Recomendação de Filmes (Content-Based)

Aplicação interativa desenvolvida com **Streamlit**, **Scikit-Learn** e **Pandas** que implementa um recomendador de filmes baseado em características de conteúdo (gêneros), combinando clusterização (**K-Means**) e ranking por **Similaridade de Cosseno vetorizada** com desempate determinístico e métricas de precisão temática.

---

## 📌 Visão Geral da Solução

O sistema analisa o catálogo do MovieLens (`movie.csv`), processa as combinações de gêneros cinematográficos e disponibiliza duas experiências integradas:
1. **Recomendador de Filmes**: O usuário seleciona um filme assistido e recebe recomendações ranqueadas por percentual de afinidade temática (**Match %** via Similaridade de Cosseno).
2. **Explorador de Dados**: Interface interativa para consultar, filtrar por múltiplos gêneros, intervalo de ano e cluster K-Means, visualizar métricas e exportar os dados em CSV.

### Pipeline de Machine Learning
1. **Limpeza e Padronização**: Extração do ano de lançamento via regex a partir do título, remoção de registros sem gênero ou com ano ausente e padronização textual em caixa alta.
2. **Engenharia de Recursos**: Codificação One-Hot (*One-Hot Encoding*) das categorias de gêneros (`Adventure`, `Comedy`, `Drama`, etc.).
3. **Clusterização (K-Means)**: Agrupamento inicial dos filmes para análise de distribuição e segmentação temático-temporal.
4. **Cálculo de Afinidade (Similaridade de Cosseno)**: Vetorização da similaridade de cosseno entre o filme de referência e todos os candidatos do catálogo (sem bloqueio rígido por cluster), ordenando por afinidade decrescente e desempate por ano mais recente.
5. **Métrica de Avaliação**: Avaliação quantitativa de sobreposição temática (`evaluate_genre_precision`).

---

## 🚀 Otimizações de Engenharia & Portfólio

Diferente de implementações ingênuas baseadas em iterações `for` e filtros repetidos em DataFrames:
- **Similaridade de Cosseno Matricial**: Cálculo em uma única operação matricial via C/BLAS com o Scikit-Learn, eliminando a distorção da distância euclidiana em dados esparsos binários e respondendo em milissegundos.
- **Eliminação do "Muro Cego"**: Busca global em todo o catálogo com opção configurável de isolamento por cluster (`same_cluster_only=False` por padrão).
- **Match Score Dinâmico**: Apresentação de percentual de afinidade visual em cada card (`95% MATCH`, `100% MATCH`).
- **Separação de Camadas**: O pipeline de preparação e treino (`src/recommender.py`) é completamente desacoplado da interface visual (`app.py`).
- **Suíte de Testes Automatizados**: Validação contínua do pré-processamento, pipeline de treinamento, ordenação de similaridade e precisão temática com `pytest`.

---

## 📁 Estrutura do Repositório

```text
├── src/
│   ├── __init__.py
│   └── recommender.py      # Limpeza, treino, similaridade de cosseno e métricas
├── tests/
│   └── test_recommender.py # Suíte de testes unitários com pytest
├── estudos/                # Cadernos de estudo originais e manuais didáticos
│   ├── codigo_aula.ipynb
│   ├── codigo_aula_executa_recsys.ipynb
│   ├── manual_deploy_app.pdf
│   └── ...
├── app.py                  # Interface web interativa em Streamlit
├── movie.csv               # Dataset de filmes
├── modelo.pkl              # Artefato persistido com dados e modelo treinado
├── pytest.ini              # Configuração de execução dos testes
├── requirements.txt        # Dependências do projeto
└── README.md
```

---

## 🛠️ Como Executar

### 1. Pré-requisitos
- Python 3.11 ou 3.12 recomendado.

### 2. Configurar o Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Executar os Testes Automatizados

```bash
pytest -v
```

### 4. Treinar e Gerar o Modelo

Caso queira reprocessar a base e retreinar o algoritmo:

```bash
python src/recommender.py
```

### 5. Iniciar a Aplicação Streamlit

```bash
streamlit run app.py
```

Acesse a interface no navegador em `http://localhost:8501`.

---

## 🧪 Testes e Qualidade

O projeto conta com testes unitários cobrindo:
- Limpeza e extração correta de datas de filmes.
- Estruturação do bundle do modelo (`BASE_DADOS`, `VARIAVEIS`, `MODELO`).
- Verificação de monotonicidade decrescente e valores normalizados `[0, 1]` da Similaridade de Cosseno.
- Desempate determinístico por ano de lançamento.
- Opção de busca global vs. restrita ao cluster.
- Cálculo de precisão de sobreposição temática (`evaluate_genre_precision`).
