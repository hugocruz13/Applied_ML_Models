# Applied ML Models

This is the second project for the *Inteligência Artificial* course at IPCA, done by Group 11. The brief was to apply three classic machine learning techniques - **Association Rules**, **Classification** and **Clustering** - each to a different real-world dataset, going through the full pipeline: exploratory data analysis, preprocessing, modelling and evaluation.

Rather than one dataset stretched three ways, we picked a dataset that actually fit each technique: Fantasy Premier League gameweek data for association rules, a body-performance dataset for classification, and Formula 1 race history for clustering. Each one lives in its own pair of notebooks - an EDA notebook that turns raw, messy data into something a model can use, and an implementation notebook that does the modelling.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![mlxtend](https://img.shields.io/badge/mlxtend-Apriori-4B8BBE)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-0.13-4C72B0)

---

## What's actually in it

- Every technique gets a full EDA pass before any model sees the data: null handling, outlier checks, univariate/bivariate analysis, and only then feature engineering - discretisation, one-hot encoding, scaling, whatever that specific dataset and algorithm needs.
- The association-rules notebook doesn't stop at one Apriori run. It's three iterations: an exploratory pass with a low `min_support` to see what the itemset space looks like, then two rounds of tightening (`min_support`, `lift`, rule length) to go from ~7,000 mostly-trivial itemsets down to a short list of rules that are actually worth reading, with a discussion of *why* each surviving rule is either expected or genuinely surprising.
- Classification isn't just "train three models and report accuracy" - each of the three (Decision Tree, Random Forest, MLP) is evaluated with a stratified train/test split, a confusion matrix, 5-fold stratified cross-validation, and then re-tuned with `RandomizedSearchCV` so the notebook can compare the hand-picked static hyperparameters against the searched ones.
- Clustering uses the elbow method to justify the choice of *k* before running K-Means, then profiles each resulting cluster on both its numeric features (points, pace, top speed, position changes...) and its categorical ones (driver, constructor, circuit, finishing status) to turn "cluster 1" into an actual driver archetype instead of just a label.
- Notebook outputs are committed cleared (see the latest commit) - the notebooks are meant to be *run*, not read as a static report of numbers. See [Known gaps](#known-gaps--possible-improvements) for what that costs.

## Techniques, datasets and notebooks

| Technique         | Dataset                                                                                                                     | EDA notebook                          | Modelling notebook                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------- |
| Association Rules | [Fantasy Premier League 2024/25](https://github.com/vaastav/Fantasy-Premier-League) gameweek data                           | `eda_association_rules_dataset.ipynb` | `implementation_association_rules.ipynb` |
| Classification    | [Body Performance Data](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)                                     | `eda_classificacao_dataset.ipynb`     | `classificacao.ipynb`                    |
| Clustering        | [Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) | `eda_clustering_dataset.ipynb`        | `clustering.ipynb`                       |

### Association Rules - Fantasy Premier League

Goal: find which player performance attributes tend to co-occur across a gameweek, using the Apriori algorithm (`mlxtend`). The EDA notebook turns raw per-gameweek stats into a binary transactional matrix - sparse counting stats (goals, cards) get binarised, continuous/skewed metrics (ICT index, BPS, threat) get tercile-discretised into Low/Medium/High via a custom `safe_qcut` helper, and rows with zero minutes played are dropped so the model isn't just learning "didn't play → didn't score."

The implementation notebook runs Apriori three times with progressively stricter thresholds and ends up with rules like `goals_scored → (threat_High, bonus)` (lift ≈ 9.5) alongside less obvious ones, e.g. goalkeepers forming an almost deterministic cluster of "starts + low creativity" (lift > 5), or assists predicting high creativity/points more strongly than the reverse.

### Classification - Body Performance

Goal: predict a person's fitness class (A-D) from body measurements and physical test results. Three models are compared - Decision Tree, Random Forest, and an MLP neural network - trained on an 80/20 stratified split, with the MLP getting `StandardScaler`-normalised inputs (fit on train only, to avoid leakage) while the tree-based models use raw features. Evaluation covers accuracy, a full classification report, confusion matrices, and 5-fold stratified CV on the Random Forest. Each model is then re-tuned with `RandomizedSearchCV` (20 iterations, 5-fold CV) so the notebook can compare the manually configured version against the searched one.

### Clustering - Formula 1

Goal: find driver/race performance archetypes from Formula 1 results, qualifying and driver data (1950-2024). Numeric features (points, laps, fastest lap time, top speed, pace, position delta, age...) are scaled and fed into K-Means after the elbow method is used to pick *k* (settled on 3). PCA reduces the result to 2D for visualisation, and each cluster is profiled against both its numeric averages and categorical make-up (driver, constructor, circuit, finishing status) - producing three fairly interpretable groups: front-runners who finish and score, mid-pack drivers who lose positions, and a "short race / early retirement" cluster dominated by mechanical failures and accidents.

## Tech stack

| Layer             | Tools                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Data handling     | pandas, numpy                                                                                                               |
| Visualisation     | matplotlib, seaborn                                                                                                         |
| Classification    | scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `MLPClassifier`), `RandomizedSearchCV`, `StratifiedKFold` |
| Clustering        | scikit-learn (`KMeans`, `PCA`, `StandardScaler`)                                                                            |
| Association rules | `mlxtend` (`apriori`, `association_rules`)                                                                                  |
| Environment       | Jupyter Notebook                                                                                                            |

## Repository layout

```
data/                              raw and preprocessed datasets used by each notebook
  bodyPerformance.csv                  classification: raw source data
  prepared_classification.csv          classification: output of the EDA notebook
  f1_dataset/                          clustering: raw F1 championship CSVs (circuits, races, results, ...)
  f1_processed_clustering.csv          clustering: output of the EDA notebook
  merged_gw.csv                        association rules: raw FPL gameweek data
  merged_gw_processed.csv              association rules: output of the EDA notebook

eda_classificacao_dataset.ipynb     EDA - Body Performance
classificacao.ipynb                 modelling - classification

eda_clustering_dataset.ipynb        EDA - Formula 1
clustering.ipynb                    modelling - K-Means clustering

eda_association_rules_dataset.ipynb EDA - Fantasy Premier League
implementation_association_rules.ipynb  modelling - Apriori / association rules

requirements.txt                    Python dependencies (see note below)
```

## Getting started

### Prerequisites

- Python 3.11+
- Jupyter Notebook or JupyterLab

### Setup

```bash
git clone git@github.com:hugocruz13/Applied_ML_Models.git
cd Applied_ML_Models

python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install mlxtend  # see note below - not currently pinned in requirements.txt
```

### Run

```bash
jupyter notebook
```

Run each pair of notebooks in order - EDA first, then the modelling notebook - since the modelling notebooks load the processed CSVs that the corresponding EDA notebook writes to `data/`.

## Known gaps / possible improvements

Being upfront about what this project doesn't do, rather than pretending it's finished:

- **`mlxtend` isn't in `requirements.txt`.** `implementation_association_rules.ipynb` hard-depends on it (`from mlxtend.frequent_patterns import apriori, association_rules`), but a plain `pip install -r requirements.txt` won't get it - a fresh clone fails on that notebook until you install it manually. This should just be added to the file.
- **No saved metrics or model artifacts.** Notebook outputs are cleared before committing, which keeps diffs clean but means there's no accuracy numbers, best hyperparameters, or plots anywhere outside of actually re-running the notebooks - nothing you can point to from a report or README without opening Jupyter. Exporting a small results table (CSV/markdown) or the tuned models (`joblib`) per technique would fix that.
- **Cluster count is picked visually, not statistically.** The elbow method plot is eyeballed to justify k=3 for K-Means; there's no silhouette score or Davies-Bouldin index backing that choice up quantitatively.
- **Association-rule thresholds were tuned by hand, iteration by iteration.** It works, and the reasoning is documented in the notebook, but it's manual - there's no systematic sweep over `min_support`/`lift` to show the trade-off curve, and no held-out data to check whether the rules generalise to a different season.
- **Hyperparameter search results aren't fed back into the models actually used later in the notebook.** `RandomizedSearchCV` runs and prints the best params/score for each classifier, but the "static" models defined earlier are what get used for the confusion matrices and feature-importance plots - the tuned versions exist only as printed output.
- **Large CSVs are committed straight into `data/`** (`merged_gw.csv` alone is ~5 MB) instead of being fetched by a small download script or `.gitignore`'d with instructions to pull them from Kaggle/GitHub - fine for a course submission, less fine for a repo people are expected to clone repeatedly.
- **No automated checks at all** - no `pytest`, no notebook execution in CI (e.g. `nbconvert --execute` or `papermill`) to catch a broken cell before it's discovered by hand.

## Academic context

Built for *Inteligência Artificial* (Licenciatura em Engenharia de Sistemas Informáticos), IPCA, 2025/26.

| Name            | Number |
| --------------- | ------ |
| Igor Costa      | 27977  |
| Gustavo Marques | 27962  |
| Gustavo Pereira | 29852  |
| Hugo Cruz       | 23010  |

