# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch implementations of ML algorithms for the "Advanced Machine Learning" course at CentraleSupélec. All code lives in Jupyter notebooks (no `.py` modules, no build system, no tests).

## Notebooks

- **Regression.ipynb** — Linear regression (OLS via pseudoinverse), Ridge (L2), Lasso (L1 with coordinate descent), Robust regression (Huber/Bisquare M-estimation). Includes a bicycle traffic prediction case study.
- **Clustering.ipynb** — Custom single-linkage agglomerative clustering (with scipy-compatible linkage matrix), plus sklearn K-Means, DBSCAN, HDBSCAN. Applied to toy datasets and NASA image segmentation.
- **Stochastic Gradient Descent.ipynb** — Logistic regression from scratch with multiple SGD optimizers (Vanilla, Momentum, Adagrad, RMSprop, ADAM, SAGA). Includes threshold tuning and hyperparameter studies.
- **Mixture Models.ipynb** — Gaussian Mixture Model (EM algorithm) implemented from scratch, compared with K-Means.
- **Graphical Models.ipynb** — Graphical Lasso (ADMM-based) and Nodewise regression for graph structure estimation. Includes AR(1)-Block and exponential decay simulations.

## Conventions

- Algorithms are implemented as classes following sklearn patterns (`fit`/`predict`/`coef_` attributes), but do NOT inherit from sklearn base classes (except `StochasticLogisticRegression` which uses `BaseEstimator`/`ClassifierMixin`).
- Design matrices are augmented with an intercept column inside `fit`/`predict` methods (using `np.c_[np.ones(...), X]`).
- Comments and markdown explanations are a mix of **French and English**.
- Core dependencies: `numpy`, `matplotlib`, `seaborn`, `sklearn`, `scipy`, `pandas`, `networkx`, `hdbscan`, `tqdm`, `Pillow`.

## Data Files

- `data.csv` — Seattle Fremont Bridge bicycle traffic (used in Regression notebook)
- `w8a.csv` — Binary classification dataset (used in SGD notebook)
- `clusterable_data.npy` — Noisy toy dataset (used in Clustering notebook)
- `im_nasa_reduced.jpg` — NASA Curiosity image (used in Clustering notebook)
