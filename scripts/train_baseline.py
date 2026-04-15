"""
Hydra-based Training Script for XGBoost on California Housing Dataset.

This module provides a configurable training pipeline for an XGBoost regression
model using Hydra for experiment management. It integrates optional experiment
tracking via Weights & Biases (W&B) and supports flexible configuration of
dataset splits, model hyperparameters, and output paths.

Key features:
    - Configuration-driven training using Hydra
    - Dataset loading via scikit-learn (California Housing)
    - Train/test split with reproducibility controls
    - XGBoost model training with customizable hyperparameters
    - Optional W&B logging for experiment tracking
    - Model checkpointing for downstream inference and evaluation

Workflow:
    1. Load configuration using Hydra
    2. Initialize W&B (if enabled)
    3. Fetch and split dataset
    4. Train XGBoost model with specified parameters
    5. Save trained model to disk
    6. Finalize experiment logging

This script is intended for:
    - Training baseline models for benchmarking
    - Generating checkpoints for JAX-based inference pipelines
    - Managing reproducible ML experiments via structured configs
"""

import os
import hydra
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import wandb
from wandb.integration.xgboost import WandbCallback

@hydra.main(config_path="../configs", config_name="xgboost_model", version_base=None)
def train(cfg: DictConfig):
    """
    Train an XGBoost regression model using Hydra configuration.

    This function orchestrates the end-to-end training pipeline:
        - Initializes experiment tracking (optional)
        - Loads and splits the dataset
        - Configures and trains an XGBoost model
        - Saves the trained model for later use

    Configuration is provided via Hydra and includes:
        - Dataset parameters (test split size, random seed)
        - XGBoost hyperparameters (depth, learning rate, estimators, etc.)
        - Logging settings (W&B enable/disable, mode)
        - Output paths for model checkpointing

    Args:
        cfg (DictConfig):
            Hydra configuration object containing:
                - dataset: parameters for data splitting
                - xgb_params: model hyperparameters
                - wandb: experiment tracking configuration
                - paths: directories and filenames for saving models
                - project_name / experiment_name: logging metadata

    Outputs:
        - Trained XGBoost model saved to disk
        - Console logs for training progress
        - Optional W&B logs for experiment tracking

    Notes:
        - Uses scikit-learn's California Housing dataset.
        - Reproducibility is controlled via `random_state` in both dataset split
        and model initialization.
        - W&B logging is enabled only if specified in the configuration.
    """
    if cfg.wandb.enabled:
        wandb.init(project=cfg.project_name, name=cfg.experiment_name, mode=cfg.wandb.mode, config=dict(cfg))

    # fetch the data from scikit learn lib
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=cfg.dataset.test_size, random_state=cfg.dataset.random_state
    )
    
    # Train XGBoost model
    print(f"Training XGBoost with max_depth={cfg.xgb_params.max_depth}...")
    model = xgb.XGBRegressor(
        n_estimators=cfg.xgb_params.n_estimators,
        max_depth=cfg.xgb_params.max_depth,
        learning_rate=cfg.xgb_params.learning_rate,
        tree_method=cfg.xgb_params.tree_method,
        objective=cfg.xgb_params.objective,
        eval_metric=cfg.xgb_params.eval_metric,
        random_state=cfg.xgb_params.random_state, 
        callbacks=[WandbCallback()] if cfg.wandb.enabled else None
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Save Model
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(cfg.paths.checkpoint_dir, cfg.paths.model_filename)
    model.save_model(save_path)
    print(f"Model saved to {save_path}")

    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    train()