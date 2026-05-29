"""
app/models/system_event.py
--------------------------
Model for records in SYSTEM.ndjson.gz.

System events describe data-feed health, gaps, and session markers.
Only fields that are stated in the data contract are modelled.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SystemEventType(str, Enum):
    GAP = "GAP"
    RECONNECT = "RECONNECT"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    HEARTBEAT = "HEARTBEAT"
    UNKNOWN = "UNKNOWN"


class SystemEvent(BaseModel):
    """
    A single record from SYSTEM.ndjson.gz.

    The only guaranteed fields from the contract are `event`, `duration`,
    and timestamp fields.  Other fields are optional.
    """

    event: SystemEventType = Field(..., description="Event classification")
    t: int = Field(..., description="Event epoch milliseconds timestamp")
    duration: Optional[float] = Field(
        default=None, description="Gap or outage duration in milliseconds"
    )
    t_start: Optional[int] = Field(default=None, description="Gap start epoch ms")
    t_end: Optional[int] = Field(default=None, description="Gap end epoch ms")
    symbol: Optional[str] = Field(default=None, description="Affected symbol if any")
    detail: Optional[str] = Field(default=None, description="Human readable detail")

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event_type(cls, v: object) -> SystemEventType:
        if isinstance(v, str):
            try:
                return SystemEventType(v.upper())
            except ValueError:
                return SystemEventType.UNKNOWN
        return v  # type: ignore[return-value]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.duration is not None:
            return self.duration / 1000.0
        return None

    @property
    def is_significant_gap(self) -> bool:
        """A gap is significant if it exceeds 5 minutes."""
        if self.duration_seconds is not None:
            return self.duration_seconds > 300
        return False
