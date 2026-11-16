
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 1. Parsed research paper
class PaperDoc(BaseModel):
    id: str
    title: str = ""
    sections: Dict[str, str] = Field(default_factory=dict)
    equations: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    profile_features: Dict[str, float] = Field(default_factory=dict)


# 2. Task sent to a specialist agent
class TaskRequest(BaseModel):
    agent_type: str
    paper_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    context_features: Dict[str, float] = Field(default_factory=dict)


# 3. Response returned by an LLM
class ModelResponse(BaseModel):
    text: str
    model_name: str
    latency: float = Field(ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)


# 4. A finding made by an agent
class Finding(BaseModel):
    type: str
    severity: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""


# 5. Final result returned by a specialist agent
class AgentResult(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    needs_human: bool = False
    raw_output: str = ""
