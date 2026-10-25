from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from .version import SERVICE_VERSION

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
Strategy = Literal['auto', 'graph', 'vector', 'hybrid']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Filters(StrictModel):
    movie: ShortText | None = None
    topic: ShortText | None = None
    sentiment: Literal['正面', '中性', '负面'] | None = None
    date_from: str | None = None
    date_to: str | None = None
    emotion: Literal['喜爱','感动','惊喜','中性','失望','愤怒','厌恶'] | None = None
    consistency: Literal['一致','矛盾','可接受'] | None = None
    min_star: int | None = Field(default=None,ge=1,le=5)
    max_star: int | None = Field(default=None,ge=1,le=5)

    @model_validator(mode='after')
    def check_dates(self):
        for value in (self.date_from, self.date_to):
            if value is not None:
                parsed = date.fromisoformat(value)
                if parsed.isoformat() != value:
                    raise ValueError('日期必须为 YYYY-MM-DD')
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError('date_from 不能晚于 date_to')
        if self.min_star is not None and self.max_star is not None and self.min_star>self.max_star:
            raise ValueError('min_star不能大于max_star')
        return self


class QueryRequest(StrictModel):
    question: Text
    strategy: Strategy = 'auto'
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Filters = Field(default_factory=Filters)
    max_cypher_rewrites: int | None = Field(default=None,ge=0,le=2)


class HistoryTurn(StrictModel):
    role: Literal['user', 'assistant']
    content: Text


class AgentRequest(StrictModel):
    question: Text
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)

    @model_validator(mode='after')
    def check_history(self):
        # A turn is a user/assistant pair. Accept at most the latest six pairs.
        for i, message in enumerate(self.history):
            expected = 'user' if i % 2 == 0 else 'assistant'
            if message.role != expected:
                raise ValueError('历史消息必须按 user、assistant 交替排列')
        if len(self.history) % 2:
            raise ValueError('历史消息必须包含完整的问答轮次')
        return self


class Evidence(BaseModel):
    id: str
    kind: Literal['aggregate', 'comment']
    source: str
    data: dict[str, Any]
    cypher: str | None = None
    score: float | None = None
    ranks: dict[str, int] = Field(default_factory=dict)


class QueryResult(BaseModel):
    request_id: str
    strategy: str
    status: Literal['ok', 'no_results', 'cannot_answer', 'insufficient_evidence', 'partial', 'error', 'timeout']
    answer: str = ''
    evidence: list[Evidence] = Field(default_factory=list)
    cypher: list[str] = Field(default_factory=list)
    subgraph: dict[str, list[dict[str, Any]]] = Field(default_factory=lambda: {'nodes': [], 'relationships': []})
    elapsed_seconds: float = 0
    model_usage: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    service_version: str = SERVICE_VERSION
    stage_timings: dict[str,float] = Field(default_factory=dict)
    evidence_checks: dict[str,Any] = Field(default_factory=dict)
    cache: dict[str,Any] = Field(default_factory=dict)
    requested_strategy: str | None = None
    routed_strategy: str | None = None
    execution_paths: list[str] = Field(default_factory=list)
    cypher_attempts: list[dict[str,Any]] = Field(default_factory=list)
    adopted_attempt: int | None = None
    graph_result_kind: Literal['rows','empty','unsupported','failed'] | None = None
    conversation: dict[str,Any] = Field(default_factory=dict)
