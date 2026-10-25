"""Explicit user-session state; summaries never mutate confirmed conditions."""
from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import QueryRequest, StrictModel

CONDITION_KEYS={'movie','topic','sentiment','date_from','date_to','emotion','consistency','min_star','max_star','top_k'}


class SessionRequest(QueryRequest):
    expected_version: int | None = Field(default=None,ge=0)
    client_request_id: str = Field(min_length=1,max_length=80,pattern=r'^[a-zA-Z0-9_-]+$')
    confirm_conditions: bool = False
    context_mode: Literal['none','full','compact'] = 'compact'


class ConditionPatch(StrictModel):
    expected_version: int = Field(ge=0)
    updates: dict[str,Any] = Field(default_factory=dict)
    unset: list[str] = Field(default_factory=list,max_length=10)

    @model_validator(mode='after')
    def known_fields(self):
        if (set(self.updates)|set(self.unset))-CONDITION_KEYS:
            raise ValueError('不支持的固定条件字段')
        if set(self.updates)&set(self.unset):raise ValueError('同一条件不能同时修改和撤销')
        if any(value is None for value in self.updates.values()):raise ValueError('撤销条件请使用unset，不要在updates里传null')
        return self


class SessionState(StrictModel):
    id: str
    version: int = 0
    created_at: float
    expires_at: float
    pinned: dict[str,Any] = Field(default_factory=dict)
    pinned_sources: dict[str,dict[str,Any]] = Field(default_factory=dict)
    candidates: dict[str,Any] = Field(default_factory=dict)
    active_conditions: dict[str,Any] = Field(default_factory=dict)
    revoked: list[str] = Field(default_factory=list)
    summary: str = ''
    turns: list[dict[str,Any]] = Field(default_factory=list)
    last_results: list[dict[str,Any]] = Field(default_factory=list)
    replay: list[dict[str,Any]] = Field(default_factory=list)


class Rewrite(StrictModel):
    standalone_question: str = Field(min_length=1,max_length=2000)
