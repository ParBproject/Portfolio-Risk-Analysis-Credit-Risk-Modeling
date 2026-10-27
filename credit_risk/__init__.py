"""Credit-risk figures for the checked-in loan book.

Dollar expected loss uses an assumed LGD. See ``credit_risk.assumptions``.
"""

__all__ = ["analyze", "load_portfolio"]


def __getattr__(name: str):
    if name in __all__:
        from credit_risk.data import load_portfolio
        from credit_risk.metrics import analyze

        return {"analyze": analyze, "load_portfolio": load_portfolio}[name]
    raise AttributeError(name)
