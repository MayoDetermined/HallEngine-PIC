"""Pakiet PIC 1D3V silnika Halla (SPT-100) z obwodem RLC, null-MCC i APR."""

from .config import Config
from .simulation import Simulation

__all__ = ["Config", "Simulation"]
