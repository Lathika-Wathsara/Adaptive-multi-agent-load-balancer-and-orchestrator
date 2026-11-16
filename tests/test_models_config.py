
from pathlib import Path
import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "models.yaml"
)


def test_model_configuration():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    models = config["models"]
    eligibility = config["task_eligibility"]

    assert len(models) == 5

    assert "qwen3:4b" in models
    assert "qwen3:14b" in models

    assert len(eligibility["general"]) == 2
    assert len(eligibility["math"]) == 3
    assert len(eligibility["vision"]) == 2

    # Verify that every eligible model exists
    for task_type, eligible_models in eligibility.items():

        assert len(eligible_models) > 0

        for model in eligible_models:
            assert model in models
            assert models[model]["enabled"] is True
            assert task_type in models[model]["supports"]


def test_serving_configuration():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    serving = config["serving"]

    assert serving["provider"] == "ollama"
    assert serving["gpu_id"] == 0
    assert serving["temperature"] == 0
