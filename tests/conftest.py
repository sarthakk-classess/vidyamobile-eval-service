"""
Shared fixtures for all test suites.
"""
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_syllabus_text():
    return (
        "Week 1: Introduction to Machine Learning\n"
        "Topics: supervised learning, linear regression, gradient descent.\n"
        "Week 2: Classification\n"
        "Topics: logistic regression, decision trees, SVMs, evaluation metrics.\n"
        "Week 3: Neural Networks\n"
        "Topics: perceptrons, backpropagation, activation functions, deep learning basics."
    )


@pytest.fixture
def sample_slide_text():
    return (
        "Slide 1: Gradient Descent\n"
        "- Iterative optimization algorithm\n"
        "- Updates θ = θ − α∇J(θ) each step\n"
        "Slide 2: Learning Rate\n"
        "- Too high: diverges. Too low: slow convergence.\n"
        "- Use learning rate schedules or adaptive methods (Adam, AdaGrad)."
    )


@pytest.fixture
def sample_academic_text():
    return (
        "Abstract: We propose a novel approach to knowledge tracing using spaced repetition. "
        "Our method combines FSRS-4.5 scheduling with difficulty estimation via gradient boosting. "
        "Experiments on 10,000 student interaction logs show a 15% improvement in retention. "
        "Conclusion: The combined system outperforms baselines on all metrics."
    )


@pytest.fixture
def student_history():
    return [
        {"timestamp": "2026-01-01T10:00:00Z", "rating": 3, "time_taken_s": 45},
        {"timestamp": "2026-01-08T10:00:00Z", "rating": 2, "time_taken_s": 90},
        {"timestamp": "2026-01-15T10:00:00Z", "rating": 4, "time_taken_s": 30},
        {"timestamp": "2026-01-22T10:00:00Z", "rating": 3, "time_taken_s": 55},
    ]


@pytest.fixture
def topic_features_raw():
    return {
        "n_reviews":         4,
        "n_lapses":          1,
        "mean_rating":       3.0,
        "min_rating":        2,
        "max_rating":        4,
        "rating_std":        0.82,
        "mean_time_taken_s": 55.0,
        "time_std":          25.0,
        "mean_interval":     7.0,
        "stability":         8.5,
        "difficulty":        6.2,
        "retrievability":    0.85,
        "review_count":      4,
        "lapse_count":       1,
        "pct_again":         0.25,
        "pct_hard":          0.25,
    }
