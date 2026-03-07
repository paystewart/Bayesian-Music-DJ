# Bayesian DJ
_A music recommendation engine that learns your taste in real time using Bayesian inference_

## Overview

Most recommendation systems treat user preferences as fixed point estimates. Bayesian DJ treats them as **probability distributions** — and updates those distributions live as you play or skip songs.

You describe a mood or scene in plain English ("late night drive, something dark and smooth"). The system parses that into **informative Gaussian priors** over nine Spotify audio features (energy, danceability, valence, etc.), then refines those priors into a posterior using **online Bayesian logistic regression with Laplace approximation** as you give feedback. Each recommendation is chosen by **Thompson sampling**, which naturally balances exploration of unfamiliar tracks against exploitation of your known preferences.

The result is a DJ that gets more accurate the more you use it — and one whose reasoning you can inspect at every step.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit UI
streamlit run user_interface.py

# Or run an interactive CLI session
python -m bayesian_dj --prompt "chill indie, low energy, acoustic"

# Run a headless strategy comparison simulation
python -m bayesian_dj --simulate --user-profile chill_listener --sim-rounds 50 --sim-repeats 20
```

The app requires a Kaggle Spotify dataset CSV (`kaggle_dataset.csv`) in the project root. Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` as environment variables to enable album art, audio previews, and listening history sync via the Spotify Web API.

---

## How It Works

### 1. Natural Language → Priors

User prompts are parsed by a two-stage pipeline. First, genre and mood labels are matched semantically using **sentence-transformers** (`all-MiniLM-L6-v2`) against a curated vocabulary. Then, matched moods and genres are mapped to **feature constraint ranges** via a hand-authored hint table (e.g., `"chill"` → `energy ∈ [0.2, 0.55]`, `acousticness ∈ [0.35, 1.0]`).

These constraint ranges are converted into the **prior mean and variance** of the Bayesian logistic regression model using:

```
μ[f] = (2 · target_f − 1) · scale
```

where `target_f` is the midpoint of the constraint range. Constrained features receive tighter variance; unconstrained features stay wide, letting the feedback loop learn them freely.

### 2. Online Bayesian Logistic Regression

The preference model is a Bayesian logistic regression over a 10-dimensional feature vector (bias + 9 audio features). The posterior over weights `β ~ N(μ, Σ)` is updated after each play/skip signal using the **Laplace approximation**:

```
Σ⁻¹ ← Σ⁻¹ + w · xxᵀ        where w = p(1 − p)
μ   ← μ + Σ · x(y − p)
```

This is a sequential Gaussian approximation to the true posterior — exact in the limit of a Gaussian likelihood, and well-behaved for logistic regression with few observations per session.

### 3. Predictive Posterior via Probit Approximation

Rather than scoring tracks with the MAP estimate alone, the system uses the **Bayesian predictive probability**, which integrates out posterior uncertainty:

```
E[σ(βᵀx)] ≈ σ(μᵀx / √(1 + π/8 · xᵀΣx))
```

This is the probit approximation to the logistic-normal integral. It produces calibrated scores that are conservative when the model is uncertain (wide posterior) and converge to the MAP estimate as the posterior tightens with more feedback.

### 4. Thompson Sampling

At each recommendation step, a weight vector is sampled from the current posterior `β̃ ~ N(μ, Σ)` and used to score the candidate pool. This gives naturally calibrated exploration — songs in uncertain regions of feature space get recommended more often early on, and the model gravitates toward high-confidence picks as it learns.

### 5. Scoring and Candidate Selection

Final song scores are a weighted combination of six signals:

| Signal | Weight | Description |
|---|---|---|
| Thompson sample score | 0.22 | Exploration via posterior sample |
| Predictive posterior score | 0.22 | Calibrated MAP-integrated estimate |
| Prior alignment score | 0.18 | Similarity to prompt-derived prior mean |
| Popularity score | 0.16 | Genre-adjusted popularity percentile |
| Taste profile bias | 0.12 | External preference signal from listening history |
| Coherence score | 0.10 | Proximity to centroid of recently liked songs |

Artist repetition and recent play history are penalized separately to maintain variety.

---

## Simulation & Evaluation

The `bayesian_dj/simulation.py` module replaces the human with a **synthetic user** defined by a known ground-truth weight vector `β*`. This enables reproducible quantitative comparison of recommendation strategies.

Four strategies are evaluated:

- **Thompson Sampling** — posterior sample-based exploration
- **Greedy (MAP)** — always picks the highest MAP-estimated song
- **Epsilon-Greedy (ε=0.1)** — greedy with 10% random exploration
- **Random** — uniform baseline

Metrics are cumulative play rate and instantaneous regret (`P(like | optimal song) − P(like | chosen song)`), averaged over 20 independent trials per strategy.

A **prior sensitivity analysis** additionally sweeps over prior scale `∈ {0.5, 1.0, 2.0, 4.0}` and constrained feature variance `∈ {0.1, 0.5, 1.0, 3.0}` under Thompson sampling, to characterize how strongly the prompt-derived prior influences convergence speed.

Two synthetic user profiles are provided:

| Profile | Characteristics |
|---|---|
| `chill_listener` | High acousticness, low energy, slightly melancholic valence, slower tempo |
| `party_lover` | High danceability and energy, positive valence, faster tempo |

---

## Diagnostics

Running with `--analyze` generates four diagnostic plots saved to `output/`:

- **Weight evolution** — posterior mean ± 95% credible interval for each feature over time
- **Prior vs. posterior density** — overlay of initial and final Gaussian densities per feature
- **Posterior entropy** — information gain over feedback rounds
- **MAP vs. predictive posterior** — scatter comparing point-estimate and uncertainty-integrated predictions, colored by play/skip outcome

---

## Project Structure

```
bayesian_dj/
├── model.py          # BayesianLogisticRegression, Laplace updates, Thompson sampling
├── session.py        # DJSession: prompt parsing, pool management, feedback loop
├── song_pool.py      # SongPool: data loading, normalization, feature matrix
├── simulation.py     # Headless strategy comparison and prior sensitivity analysis
├── diagnostics.py    # Posterior diagnostic plots
music_query_parser/
├── parser.py         # MusicQueryParser: NLP → QuerySpec with genre/mood/constraint extraction
├── embedder.py       # SemanticEmbedder: sentence-transformers with TF-IDF fallback
user_interface.py     # Streamlit UI with Spotify integration
```

---

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `matplotlib`, `streamlit`  
NLP: `sentence-transformers`, `scikit-learn` (TF-IDF fallback)  
Visualization: `plotly` (optional, falls back to Streamlit native charts)  
Data: Kaggle Spotify dataset (`kaggle_dataset.csv`)

---

## Design Notes

**Why Laplace approximation?** It keeps the posterior update closed-form and O(d²) per observation — appropriate for an interactive session where the model must update in real time after each play/skip signal. Variational inference or MCMC would be more expressive but impractical here.

**Why Thompson sampling over greedy?** Greedy MAP selection collapses to exploitation immediately — it stops exploring unfamiliar tracks once it has any posterior signal. Thompson sampling maintains calibrated uncertainty across the session, which matters most in the first 5-10 songs when the posterior is still diffuse.

**Why the probit approximation?** The MAP prediction `σ(μᵀx)` ignores posterior variance, which leads to overconfident scores for songs in unexplored regions of feature space. The probit correction penalizes high-variance predictions, effectively asking "how confident am I that this is a good match?" rather than just "what's my best guess?"
