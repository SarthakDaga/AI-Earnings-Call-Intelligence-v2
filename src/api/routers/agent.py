import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import Services, get_services
from src.api.schemas import AgentRequest

router = APIRouter(prefix="/v1/agent", tags=["agent"])


@router.post("/chat")
def chat(req: AgentRequest, svc: Services = Depends(get_services)):
    """Server-Sent Events: route, thought, tool_call, tool_result, verify, answer, done."""
    def gen():
        for event in svc.agent.stream(req.message, req.thread_id):
            yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
