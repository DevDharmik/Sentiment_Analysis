# YouTube Comment Sentiment Analysis

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Compare classical ML and modern transformer models for sentiment analysis on real YouTube comments. Fetches comments live via the YouTube Data API, runs four models in parallel, and visualises the results in an interactive Streamlit dashboard.

This project is a methodologically honest rebuild and extension of an undergraduate paper that reported 91% accuracy for Linear SVM + TF-IDF on YouTube comments. Here the classical baseline is reconstructed transparently (with seed and hyperparameters documented), then directly compared against a modern transformer to show where each approach actually holds up.

🚧 **Live demo:** Coming soon on Streamlit Cloud.

---

## Stack

- **Data:** YouTube Data API v3
- **Models:** scikit-learn (LinearSVC, LogisticRegression, MultinomialNB) with TF-IDF; HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Dashboard:** Streamlit + Plotly
- **Development:** GitHub Codespaces, Google Colab

## Features

- On-demand sentiment analysis on any YouTube video (positive / neutral / negative)
- Side-by-side comparison of 4 models on the same comments
- Engagement-weighted sentiment score that gives more weight to liked comments
- Top liked positive and negative comments surfaced automatically
- Sentiment evolution over time for the video's comment thread
- SVM token attribution — see exactly which words drove each linear-model prediction
- One-click CSV export of all analysed comments and their predictions

---

## Project structure

```
.
├── src/
│   ├── extractor.py          # YouTube Data API v3 wrapper
│   ├── preprocessing.py      # Text cleaning, demoji, langdetect
│   ├── sentiment.py          # HuggingFace transformer scorer
│   ├── train_classical.py    # Train SVM / LogReg / NB (one-off)
│   ├── cache.py              # SQLite cache for videos + predictions
│   └── explain.py            # Token-level attribution for linear models
├── notebooks/                # Colab exploration notebooks
├── tests/                    # pytest unit tests
├── data/                     # SQLite cache (gitignored)
├── models/                   # Trained .joblib artifacts (gitignored)
├── outputs/                  # CSV exports (gitignored)
├── streamlit_app.py          # Dashboard entry point
├── requirements.txt
├── .devcontainer/            # GitHub Codespaces config
└── .streamlit/               # Dashboard config + secrets template
```

---

## Quick start

### 1. Get a YouTube Data API v3 key

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → create a project
2. Enable **YouTube Data API v3**
3. Create credentials → API key → copy it

### 2. Clone and install

```bash
git clone https://github.com/DevDharmik/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install -r requirements.txt
```

### 3. Configure your API key

For local development:
```bash
cp .env.example .env
# Edit .env and paste your YouTube API key
```

For Streamlit (local or Cloud):
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and paste your key
```

### 4. Train the classical models (one-off, ~30 seconds)

```bash
python -m src.train_classical
```

This downloads the Pang & Lee `movie_reviews` corpus from NLTK and trains LinearSVC, LogisticRegression, and MultinomialNB on TF-IDF features. Outputs go to `models/`.

### 5. Run the dashboard

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), paste any YouTube URL, and analyse.

---

## Running in GitHub Codespaces

Click **Code → Codespaces → Create codespace on main**. The devcontainer installs dependencies automatically. Add your API key to Codespaces Secrets (`YOUTUBE_API_KEY`) and run:

```bash
streamlit run streamlit_app.py
```

The port 8501 is auto-forwarded.

## Running in Google Colab

Open `notebooks/01_pipeline_exploration.ipynb`, add `YOUTUBE_API_KEY` (and optionally `GITHUB_PAT`) to Colab Secrets via the 🔑 icon, and run cells top to bottom.

---

## How it works

1. **Extraction** — `src/extractor.py` parses the video ID from any YouTube URL and pulls metadata + comments via the official API, with `tenacity`-backed retries.
2. **Preprocessing** — `src/preprocessing.py` strips HTML, URLs, normalises whitespace, replaces emojis with their text descriptions, and detects language.
3. **Classical prediction** — Pre-trained LinearSVC, LogisticRegression, and MultinomialNB models (loaded from `models/`) score each comment via TF-IDF features.
4. **Transformer prediction** — `cardiffnlp/twitter-roberta-base-sentiment-latest` scores each comment with `(label, confidence)`. The transformer is trained on tweets, which match YouTube comment style far better than movie reviews — making it the most reliable predictor in the pipeline.
5. **Caching** — All predictions are written to a local SQLite database in `data/`. Re-analysing the same video is instant.
6. **Explainability** — For any linear classical model, `src/explain.py` decomposes a prediction into per-token contributions (`tfidf × coefficient`) so users can see exactly which words drove a prediction.
7. **Dashboard** — `streamlit_app.py` ties it all together: distribution charts, model agreement, engagement-weighted scoring, top comments, sentiment-over-time, and CSV export.

---

## Methodology notes

- **Why include classical models if they underperform?** The classical pipeline is the v1 baseline — it shows what TF-IDF + linear models can and cannot do, and it powers the explainability feature (linear models give exact, fast token attribution that transformers cannot match without approximation).
- **Why the transformer is trusted on YouTube comments.** It was fine-tuned on tweets, which share the short, informal, emoji-heavy style of YouTube comments. The classical models were trained on movie reviews, so they exhibit measurable domain mismatch when deployed on social text — and the dashboard surfaces this disagreement openly rather than hiding it.
- **Accuracy claims.** Classical models report ~84–85% macro-F1 on the Pang & Lee held-out test set. The transformer's published accuracy on its training distribution is ~94% (TweetEval). No accuracy is claimed on YouTube comments specifically, as there is no hand-labelled YouTube test set in this project.

---

## Testing

```bash
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Dharmik Champaneri** — M.Sc. Data Science, University of Europe for Applied Sciences
[GitHub](https://github.com/DevDharmik) · [LinkedIn](https://www.linkedin.com/in/dharmikchampaneri/)
