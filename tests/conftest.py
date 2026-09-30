import pytest

from credit_risk.data import DATA_PATH, load_portfolio
from credit_risk.metrics import analyze


@pytest.fixture(scope="session")
def portfolio():
    frame = load_portfolio(DATA_PATH)
    return frame, analyze(frame, DATA_PATH)
