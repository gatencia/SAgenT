import dataclasses
import os
import json
from typing import Dict, Any

@dataclasses.dataclass
class IRConfig:
    backend: str
    backend_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_env_or_file() -> 'IRConfig':
        # 1. Try Env Var
        env_backend = os.environ.get("IR_BACKEND")
        if env_backend:
            return IRConfig(backend=env_backend)
        
        # 2. Try Config Path
        config_path = os.environ.get("IR_CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return IRConfig(
                        backend=data.get("backend", "cnf"),
                        backend_params=data.get("backend_params", {})
                    )
            except Exception:
                pass
        
        # Default
        return IRConfig(backend="pb")
