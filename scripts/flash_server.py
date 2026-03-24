#!/usr/bin/env python3
import time
import json
import uuid
import sys
import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from mlx_flash import FlashConfig, FlashGenerationLoop

app = FastAPI(title="mlx-flash Inference Server")

# Global state
model_loop: Optional[FlashGenerationLoop] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    print("[⚡ mlx-flash] Server starting... will load model on first request or as specified via args.")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model_loop
    
    # Lazy load or verify model path
    if model_loop is None:
        # For this prototype, we'll use the 30B path from our tests
        model_path = "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"
        print(f"[⚡ mlx-flash] Loading model: {model_path}")
        config = FlashConfig(ram_budget_gb=4.0, debug=True)
        model_loop = FlashGenerationLoop(model_path, config=config)
    
    # Simple prompt formatting (Llama-3 style)
    prompt = ""
    for msg in request.messages:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    async def generate_chunks():
        try:
            for text_chunk in model_loop.stream_generate(
                prompt, 
                max_tokens=request.max_tokens or 512, 
                temp=request.temperature or 0.7
            ):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"[!] Generation error: {e}", file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    if request.stream:
        return StreamingResponse(generate_chunks(), media_type="text/event-stream")
    
    # Non-streaming (collected)
    full_text = ""
    for text_chunk in model_loop.stream_generate(prompt, max_tokens=request.max_tokens, temp=request.temperature):
        full_text += text_chunk
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[⚡ mlx-flash] OpenAI-Compatible Server listening on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
