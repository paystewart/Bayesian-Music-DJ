# Bayesian Music Recommendation Engine
_Utilizing Bayesian Machine Learning for Dynamic Playlists_

## Launch the UI

Run the Streamlit app locally from the project root:

```bash
streamlit run user_interface.py
```

If the app reports missing packages, install the project dependencies first:

```bash
pip install -r requirements.txt
```

## Project Overview
Our team built an improved music recommendation engine that moves away from static point-based systems. By framing recommendation as a Bayesian Inference problem, we can treat user preferences and track metadata as probability distributions. This allows the system to update the likelihood of a song's relevance in real-time based on natural language prompts.

## Methodology
Our approach integrates deep learning embeddings with a Bayesian curriculum to ensure realistic recommendations.

**1. The Model: Sentence-Transformers**

We utilize the pre-trained all-MiniLM-L6-v2 model from Hugging Face to embed user prompts and Spotify metadata.

- Latent Space: These embeddings are mapped into a shared high-dimensional Gaussian latent space.
- Bayesian Priors: We treat these initial embeddings as prior distributions rather than fixed vectors.

**2. Bayesian Ranking & Inference**

To generate the final playlist, the system follows a probabilistic process:

- Similarity Search: Retrieves a candidate pool from the Spotify dataset.
- Bayesian Decision Theory: This layer maximizes the expected utility for the user’s specific mood or seed-song descriptor.
- Ranking Layer: A final Bayesian Ranking layer orders the tracks to capture relative preference strengths between genres and moods.

## Data Source

We are utilizing real-world semi-structured JSON data sourced from the Spotify Web API.

- Scope: Thousands of tracks including artist, genre, release year, and high-level audio features.
- Hierarchical Structure: Following the principles of Hierarchical Models, we group tracks by genre while maintaining individual song characteristics.
