from typing import Optional
from engine.connectivity.base import ConnectivityEncoder
from engine.connectivity.rank_tree import RankTreeConnectivity

class ConnectivityRegistry:
    def __init__(self):
        self._encoders = {}
        self.register(RankTreeConnectivity())

    def register(self, encoder: ConnectivityEncoder):
        self._encoders[encoder.name] = encoder

    def get_supported_kind(self, kind: str) -> Optional[ConnectivityEncoder]:
        for enc in self._encoders.values():
            if kind in enc.required_kinds():
                return enc
        return None
