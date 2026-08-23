# liveness와 readiness가 공개할 수 있는 상태 문자열을 최소 계약으로 제한한다.
"""Public response schemas for Kubernetes-style service probes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


# HTTP 프로세스 생존 여부만 나타내므로 허용 값은 UP 하나뿐이다.
class LiveStatusResponse(BaseModel):
    """Liveness response; it does not describe model readiness."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["UP"]


# 모델 레지스트리 준비 여부를 READY 또는 NOT_READY로만 표현한다.
class ReadyStatusResponse(BaseModel):
    """Readiness response derived only from the model registry state."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "NOT_READY"]
