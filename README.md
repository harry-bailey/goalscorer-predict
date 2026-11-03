# FPL Goalscorer Prediction

A machine learning project exploring the prediction of Premier League goalscorers using data from the Fantasy Premier League (FPL) API.

The project uses historical player performance, fixture information and engineered statistical features to train a binary classification model that estimates the probability of a player scoring in their next fixture.

## Project Structure

- `notebooks/01_data_preparation.ipynb` – Data acquisition, cleaning and feature engineering.
- `notebooks/02_model_training_evaluation.ipynb` – Model training, performance evaluation and visualisation.
- `live_inference.py` – Automated data acquisition and weekly prediction pipeline.

## Methodology

The project involves:
- Acquiring historical player and fixture data through the FPL API.
- Cleaning and merging datasets.
- Engineering player-level features, including historical performance statistics and rolling averages.
- Training machine learning classification models.
- Evaluating predictive performance using metrics such as ROC-AUC and Precision-Recall AUC.

## Current Status

The initial data preparation and modelling pipeline has been developed. Further work will explore alternative models, additional predictive features and improvements to model performance.

## Data Source

Data is obtained from the [Fantasy Premier League API](https://fantasy.premierleague.com/api/)
and https://github.com/vaastav/Fantasy-Premier-League
