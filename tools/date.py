import json
from agents import  RunContextWrapper, function_tool
from dataclass.research import ResearchContext

@function_tool
async def get_current_date_info(wrapper: RunContextWrapper[ResearchContext]) -> str:
    return json.dumps({
        "current_date": wrapper.context.current_date,
        "current_year": wrapper.context.current_year
    })
