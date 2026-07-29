"""Public response schemas for Kubernetes-style service probes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LiveStatusResponse(BaseModel):
    """Liveness response; it does not describe model readiness."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["UP"]


class ReadyStatusResponse(BaseModel):
    """Readiness response derived only from the model registry state."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "NOT_READY"]
