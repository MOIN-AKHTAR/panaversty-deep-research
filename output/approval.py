from pydantic import BaseModel

class ApproveAgentResponse(BaseModel):
    approved: bool
    query: str
    rejection_reason: str
    final_result: str