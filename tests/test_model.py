import numpy as np
import pandas as pd

from src.model_trainer import ModelTrainer


def test_regression_training_saves_best_model(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.uniform(0, 1, size=(80, 4)),
        columns=["accident", "speed", "lighting", "crossing"],
    )
    y = (
        0.45 * X["accident"]
        + 0.30 * X["speed"]
        - 0.15 * X["lighting"]
        + 0.20 * X["crossing"]
    ).clip(0, 1)

    trainer = ModelTrainer(random_state=42)
    results = trainer.train_regression(
        X, y, save_path=tmp_path / "best_model.pkl"
    )

    assert trainer.best_model is not None
    assert trainer.best_model_name in results
    assert (tmp_path / "best_model.pkl").exists()
    assert results[trainer.best_model_name]["R2"] > 0.5

