# Sentiment Analysis on YouTube Comments

A 2026 rebuild of my 2023 undergraduate research paper, comparing classical machine learning methods (SVM, Logistic Regression, Naive Bayes) with a modern transformer on real YouTube comment data.

## What it does

Paste any YouTube video URL → get sentiment analysis of the comments using four AI models side-by-side. See which models agree, which disagree, and why.

## The story

- **v1 (2023)** — University of Mumbai research paper. Used SVM + TF-IDF on movie reviews as a proxy for YouTube comments. Reported 91% F1.
- **v2 (2026, this repo)** — Real YouTube comments via the Data API. Same SVM + 2 alternative classical models trained on the same dataset with documented hyperparameters. Plus a transformer (Twitter-RoBERTa, 124M tweet-trained) for comparison.

Honest finding: with documented settings, the classical baseline lands at 84–85% — not 91%. The 2023 paper's numbers likely came from undocumented hyperparameter choices.

## Stack

- **Data:** YouTube Data API v3
- **Classical ML:** scikit-learn (LinearSVC, LogisticRegression, MultinomialNB) + TF-IDF
- **Transformer:** HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Preprocessing:** `langdetect`, `demoji`
- **Dashboard:** Streamlit + Plotly
- **Dev:** GitHub Codespaces + Google Colab

## Features

- Real-time sentiment analysis (positive / neutral / negative)
- Side-by-side comparison of 4 AI models
- Engagement-weighted sentiment score (likes count more)
- Top liked positive + negative comments
- Sentiment evolution over time
- Word clouds per sentiment
- SVM token attribution — see which words drove each prediction
- CSV export of analysed comments

## Live demo

🚧 Coming soon — to be deployed on Streamlit Cloud.

## Authors

- **v1 (2023):** Aniruddha Mestry, Siddiqui M. Zakir, Dharmik Champaneri — University of Mumbai
- **v2 (2026):** Dharmik Champaneri — M.Sc. Data Science, University of Europe for Applied Sciences (Berlin)

## License

MIT
