import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from hefty import(
    solar,
    pv_model,
    utilities,
    wind,
    custom,
)
