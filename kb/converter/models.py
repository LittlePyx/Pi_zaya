from __future__ import annotations

from typing import Optional, Callable
from pydantic import BaseModel, Field

class TextBlock(BaseModel):
    bbox: tuple[float, float, float, float]
    text: str
    max_font_size: float = 0.0
    is_bold: bool = False
    is_code: bool = False
    is_table: bool = False
    table_markdown: Optional[str] = None
    is_math: bool = False
    is_caption: bool = False  # Figure/Table captions
    heading_level: Optional[str] = None  # e.g., "[H1]", "[H2]"
    
    # Optional closure for inserting images (not serializable, exclude from dumps if needed)
    insert_image: Optional[Callable] = Field(default=None, exclude=True)
