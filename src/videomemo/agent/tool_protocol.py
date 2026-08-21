from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class ToolResponse:
    """Serializable response envelope for every agent tool call."""

    status: ToolStatus
    data: Any = None
    message: str = ""
    error_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != ToolStatus.ERROR

    def dump(self) -> dict:
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "error_code": self.error_code,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def success(cls, data: Any, message: str = "", **metadata: Any) -> "ToolResponse":
        return cls(ToolStatus.SUCCESS, data=data, message=message, metadata=metadata)

    @classmethod
    def partial(cls, data: Any, message: str, **metadata: Any) -> "ToolResponse":
        return cls(ToolStatus.PARTIAL, data=data, message=message, metadata=metadata)

    @classmethod
    def error(cls, code: str, message: str, **metadata: Any) -> "ToolResponse":
        return cls(
            ToolStatus.ERROR,
            data=None,
            message=message,
            error_code=code,
            metadata=metadata,
        )
