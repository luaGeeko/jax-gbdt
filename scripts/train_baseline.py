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