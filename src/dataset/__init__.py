"""Dataset loaders and preparation pipelines."""

__all__ = [
    "ALL_TASTE_LABELS", "LOW_SHOT_TASTE_LABELS", "ODOR_FAMILIES",
    "SALT_LABEL", "SOUR_LABEL", "TASTE_LABELS",
]


def __getattr__(name: str):
    """Expose sensory constants without pre-importing its CLI module.

    Importing it eagerly makes ``python -m src.dataset.sensory`` trigger
    runpy's duplicate-module warning in Colab.
    """
    if name in __all__:
        from . import sensory
        return getattr(sensory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
