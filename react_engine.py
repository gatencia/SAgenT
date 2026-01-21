# Compatibility Shim for legacy imports
# The monolithic react_engine.py has been refactored into the 'engine' package.

from engine import ReActAgent, AgentState, IRConfig
from engine.actions import ActionType
from engine.state import ModelingConstraint
from engine.sat_manager import SATManager
from engine.backends.base import IRBackend
from engine.backends.registry import IRBackendRegistry
from engine.connectivity.base import ConnectivityEncoder
from engine.connectivity.registry import ConnectivityRegistry

# Logic is now in engine/
