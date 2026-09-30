"""Portfolio credit-risk calculations for the checked-in loan book.

The public figures live in :func:`credit_risk.metrics.analyze`. Dollar
expected loss needs an LGD, which this file does not contain. See
``credit_risk.assumptions``.
"""

__all__ = ["analyze", "load_portfolio"]


def __getattr__(name: str):
    if name in __all__:
        from credit_risk.data import load_portfolio
        from credit_risk.metrics import analyze

        return {"analyze": analyze, "load_portfolio": load_portfolio}[name]
    raise AttributeError(name)
