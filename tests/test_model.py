
from src.model import train_model


def test_model_trains_successfully():
    model, accuracy = train_model()

    assert model is not None
    assert 0.0 <= accuracy <= 1.0


def test_model_meets_minimum_accuracy():
    _, accuracy = train_model()

    # Example threshold for the Iris dataset
    assert accuracy >= 0.80
