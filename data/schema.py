from pydantic import BaseModel, Field
from typing import List, Optional

class BugReport(BaseModel):
    user_name: Optional[str] = Field(description="Name of the user reporting the issue")
    os_version: str = Field(description="Operating system and version, e.g., iOS 17.2")
    device_model: str = Field(description="Hardware device, e.g., iPhone 14 Pro")
    issue_type: str = Field(description="Categorize as: crash, UI_glitch, latency, or feature_request")
    reproduction_steps: List[str] = Field(description="Step-by-step actions that caused the bug")
