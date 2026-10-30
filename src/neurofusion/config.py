from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Config:
    memory_dim: int = 256
    context_dim: int = 128
    robot_dim: int = 16
    frame_feature_dim: int = 128
    event_feature_dim: int = 64
    action_dim: int = 5
    control_hz: int = 50
    max_pending_messages: int = 4096

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{field.name} must be a positive integer')
        if self.action_dim != 5:
            raise ValueError('The initial named action schema has exactly five dimensions')

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open('rb') as stream:
            return cls(**tomllib.load(stream))


def vector(value, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in value)
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f'{name} must contain {size} finite values')
    return result
