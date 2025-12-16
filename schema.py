from pydantic import BaseModel
from typing import List, Any

class PredictSchema(BaseModel):
    data: List[List[Any]]