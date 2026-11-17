"""Extra structures produced by ingestion (B1).

`ParsedPaper` IS-A `PaperDoc`: every field of the frozen schema is kept as-is, so
any code that expects a PaperDoc keeps working. The extras (equation numbers and
context, figure captions, ...) are additive and have defaults. If Person A wants
them in schemas.py, promote them through a PR both approve.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from src.common.schemas import PaperDoc


class EquationInfo(BaseModel):
    index: int
    number: Optional[str] = None      # "3", "2.1", or None if unnumbered
    latex: str                        # LaTeX if source="tex", flattened text if "pdf"
    label: Optional[str] = None
    section: str = ""
    before: str = ""                  # derivation text just before the equation
    after: str = ""                   # and just after it
    source: str = "pdf"               # "tex" | "pdf"


class FigureInfo(BaseModel):
    id: str                           # "Figure 3"
    path: str                         # cropped PNG, relative to the repo root
    page: int
    caption: str
    citing_paragraphs: List[str] = Field(default_factory=list)
    fallback_crop: bool = False       # True if no graphic was found (blind crop)


class TableInfo(BaseModel):
    id: str                           # "Table 2"
    page: int
    caption: str
    text: str = ""                    # rows joined by newline, cells by " | "
    citing_paragraphs: List[str] = Field(default_factory=list)


class ParsedPaper(PaperDoc):
    arxiv_id: str = ""
    category: str = ""                # theory | empirical | mixed
    n_pages: int = 0
    equation_source: str = "pdf"      # "tex" | "pdf"
    equation_meta: List[EquationInfo] = Field(default_factory=list)
    figure_meta: List[FigureInfo] = Field(default_factory=list)
    table_meta: List[TableInfo] = Field(default_factory=list)

    def to_paperdoc(self) -> PaperDoc:
        data = {k: getattr(self, k) for k in PaperDoc.model_fields} if hasattr(
            PaperDoc, "model_fields"
        ) else {k: getattr(self, k) for k in PaperDoc.__fields__}
        return PaperDoc(**data)


def to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):          # pydantic v2
        return model.model_dump_json(indent=2)
    return model.json(indent=2)                    # pydantic v1


def from_json(cls, text: str):
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(text)
    return cls.parse_raw(text)
