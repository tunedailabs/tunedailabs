"""
TunedAI Reasoning Engine - API Server
======================================
OpenAI-compatible endpoint that adds structured causal reasoning (T1-T4)
to any LLM. Drop-in replacement for Knapsack's current model endpoint.

Architecture:
    Knapsack Agent --> POST /v1/chat/completions --> Local Model + Reasoning LoRA --> T1-T4 Response

Backends (auto-detected):
    1. Local tuned model via Ollama (preferred - zero API dependency)
    2. OpenAI API with reasoning system prompt (demo fallback)

Usage:
    python knapsack_api.py
    # Runs on http://localhost:9001

Endpoints:
    POST /v1/chat/completions            Reasoning engine (T1-T4 structured output)
    POST /v1/chat/completions/baseline   Generic model (no reasoning - for comparison)
    GET  /v1/models                      Available models
    GET  /health                         Health check
"""

import os
import json
import time
import uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="TunedAI Reasoning Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# The reasoning instruction that the tuned model has baked into weights.
# When using a generic model, we inject this as a system prompt.
# When using the tuned model, this is unnecessary (behavior is intrinsic).
REASONING_SYSTEM_PROMPT = """You are an executive reasoning engine. For every request, produce structured causal analysis using exactly four tiers:

TIER 1 — Observation:
State the facts as given. Identify patterns, anomalies, and missing data. Do not interpret yet — just observe precisely.

TIER 2 — Mechanism:
Explain WHY things are the way they are. Identify root causes, causal chains, and driving forces. Connect observations to underlying dynamics.

TIER 3 — Projection:
Project what happens if the current trajectory continues. What are the consequences of action vs. inaction? Give specific, concrete projections — not vague warnings.

TIER 4 — Simulation:
Run a counterfactual. What would have happened if a different decision had been made? What alternative action produces a better outcome? Be specific about the comparison.

Rules:
- Always produce all four tiers unless the question is genuinely simple enough to answer in one sentence.
- When you don't have enough data to answer accurately, say so explicitly. Do not fabricate. State what data you need.
- When evaluating information from multiple sources, assess each source's credibility: methodology, incentive structures, recency, independence. Flag vendor content. Weight independent analysis higher.
- Be direct and concise. Executives read this. No filler."""

# Knapsack agent personas
AGENT_PERSONAS = {
    "polly": {
        "name": "Polly",
        "role": "Inbox & Social Monitor",
        "instruction": "You are Polly, the user's inbox and social media monitor. You are warm, concise, and perceptive. Apply structured reasoning to email triage, priority assessment, and communication pattern analysis."
    },
    "scout": {
        "name": "Scout",
        "role": "Executive Assistant",
        "instruction": "You are Scout, the user's executive assistant. You are organized, proactive, and detail-oriented. Apply structured reasoning to calendar optimization, meeting prep, task tracking, and time management."
    },
    "atlas": {
        "name": "Atlas",
        "role": "Relationship Optimizer",
        "instruction": "You are Atlas, the user's relationship optimizer. You are strategic, insightful, and opportunity-driven. Apply structured reasoning to relationship risk assessment, networking strategy, and business development."
    },
    "coach": {
        "name": "Coach",
        "role": "Work Coach",
        "instruction": "You are Coach, the user's daily work coach. You are direct, analytical, and encouraging. Apply structured reasoning to productivity analysis, work patterns, context switching, and habit formation."
    },
}


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

async def check_ollama():
    """Check if Ollama is running and has a model available."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                return len(models) > 0
    except Exception:
        pass
    return False


async def check_openai():
    """Check if OpenAI API key is configured."""
    return bool(OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

async def infer_ollama(messages: list, model: str = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    """Run inference through Ollama."""
    model = model or OLLAMA_MODEL
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


async def infer_openai(messages: list, model: str = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    """Run inference through OpenAI API."""
    model = model or OPENAI_MODEL
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def infer(messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> tuple[str, str]:
    """Route to the best available backend. Returns (response_text, backend_used)."""
    if await check_ollama():
        return await infer_ollama(messages, temperature=temperature, max_tokens=max_tokens), "ollama"
    elif await check_openai():
        return await infer_openai(messages, temperature=temperature, max_tokens=max_tokens), "openai"
    else:
        raise RuntimeError("No inference backend available. Start Ollama or set OPENAI_API_KEY.")


# ---------------------------------------------------------------------------
# Build message list with reasoning instruction
# ---------------------------------------------------------------------------

def build_reasoning_messages(messages: list, agent: str = None) -> list:
    """Inject reasoning system prompt and optional agent persona."""
    system_parts = [REASONING_SYSTEM_PROMPT]

    if agent and agent.lower() in AGENT_PERSONAS:
        persona = AGENT_PERSONAS[agent.lower()]
        system_parts.append(f"\n{persona['instruction']}")

    # Check if there's already a system message
    has_system = any(m.get("role") == "system" for m in messages)

    if has_system:
        # Prepend reasoning instruction to existing system message
        result = []
        for m in messages:
            if m["role"] == "system":
                result.append({
                    "role": "system",
                    "content": "\n\n".join(system_parts) + "\n\n" + m["content"]
                })
            else:
                result.append(m)
        return result
    else:
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + messages


# ---------------------------------------------------------------------------
# OpenAI-compatible API endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions with reasoning engine."""
    body = await request.json()
    messages = body.get("messages", [])
    agent = body.get("agent", None)  # Extension: specify Knapsack agent
    temperature = body.get("temperature", 0.3)
    max_tokens = body.get("max_tokens", 2048)

    reasoning_messages = build_reasoning_messages(messages, agent=agent)

    start = time.time()
    content, backend = await infer(reasoning_messages, temperature=temperature, max_tokens=max_tokens)
    elapsed = time.time() - start

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"tunedai-reasoning-7b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "").split()) for m in reasoning_messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.get("content", "").split()) for m in reasoning_messages) + len(content.split()),
        },
        "tunedai": {
            "backend": backend,
            "inference_time_ms": round(elapsed * 1000),
            "agent": agent,
            "reasoning_engine": True,
        }
    })


@app.post("/v1/chat/completions/baseline")
async def chat_completions_baseline(request: Request):
    """Baseline model without reasoning engine — for comparison."""
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 1024)

    start = time.time()
    content, backend = await infer(messages, temperature=temperature, max_tokens=max_tokens)
    elapsed = time.time() - start

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"baseline-{OLLAMA_MODEL}",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(content.split()),
        },
        "tunedai": {
            "backend": backend,
            "inference_time_ms": round(elapsed * 1000),
            "reasoning_engine": False,
        }
    })


@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = []
    if await check_ollama():
        models.append({
            "id": "tunedai-reasoning-7b",
            "object": "model",
            "owned_by": "tunedai-labs",
            "meta": {"backend": "ollama", "model": OLLAMA_MODEL, "local": True},
        })
    if await check_openai():
        models.append({
            "id": "tunedai-reasoning-cloud",
            "object": "model",
            "owned_by": "tunedai-labs",
            "meta": {"backend": "openai", "model": OPENAI_MODEL, "local": False},
        })
    return JSONResponse({"object": "list", "data": models})


@app.get("/health")
async def health():
    """Health check with backend status."""
    ollama_ok = await check_ollama()
    openai_ok = await check_openai()
    return JSONResponse({
        "status": "ok" if (ollama_ok or openai_ok) else "no_backend",
        "backends": {
            "ollama": {"available": ollama_ok, "url": OLLAMA_URL, "model": OLLAMA_MODEL},
            "openai": {"available": openai_ok, "model": OPENAI_MODEL},
        },
        "version": "1.0.0",
    })


# ---------------------------------------------------------------------------
# Simplified agent endpoint (for the demo frontend)
# ---------------------------------------------------------------------------

@app.post("/api/agent")
async def agent_endpoint(request: Request):
    """Simplified endpoint for the demo frontend.

    Body: {"agent": "polly|scout|atlas|coach", "message": "user's question"}
    Returns: {"agent": "...", "response": "...", "tiers": [...], "time_ms": ...}
    """
    body = await request.json()
    agent_name = body.get("agent", "atlas")
    user_message = body.get("message", "")

    if not user_message.strip():
        return JSONResponse({"error": "Message is required"}, status_code=400)

    messages = [{"role": "user", "content": user_message}]
    reasoning_messages = build_reasoning_messages(messages, agent=agent_name)

    start = time.time()
    content, backend = await infer(reasoning_messages, temperature=0.3, max_tokens=2048)
    elapsed = time.time() - start

    # Parse tiers from response
    tiers = parse_tiers(content)

    persona = AGENT_PERSONAS.get(agent_name.lower(), {})

    return JSONResponse({
        "agent": persona.get("name", agent_name),
        "role": persona.get("role", ""),
        "response": content,
        "tiers": tiers,
        "backend": backend,
        "time_ms": round(elapsed * 1000),
        "reasoning_engine": True,
    })


@app.post("/api/agent/baseline")
async def agent_baseline_endpoint(request: Request):
    """Baseline agent response for comparison."""
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message.strip():
        return JSONResponse({"error": "Message is required"}, status_code=400)

    messages = [{"role": "user", "content": user_message}]

    start = time.time()
    content, backend = await infer(messages, temperature=0.7, max_tokens=1024)
    elapsed = time.time() - start

    return JSONResponse({
        "response": content,
        "backend": backend,
        "time_ms": round(elapsed * 1000),
        "reasoning_engine": False,
    })


def parse_tiers(text: str) -> list:
    """Extract tier blocks from model output."""
    import re
    tiers = []
    # Match patterns like "TIER 1 — Observation:" or "TIER 1 - Observation:"
    pattern = r'TIER\s+(\d)\s*[—\-–]\s*([\w\s]+?):\s*\n?(.*?)(?=TIER\s+\d|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    for num, name, body in matches:
        tiers.append({
            "tier": int(num),
            "name": name.strip(),
            "content": body.strip(),
        })
    return tiers


# ---------------------------------------------------------------------------
# Serve static files (demo pages)
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>TunedAI Reasoning Engine</h1><p>API running. See /health for status.</p>")


@app.get("/agent")
async def serve_agent():
    agent_path = os.path.join(STATIC_DIR, "knapsack_agent.html")
    if os.path.exists(agent_path):
        with open(agent_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Agent interface not found</h1>")


@app.get("/demo")
async def serve_demo():
    demo_path = os.path.join(STATIC_DIR, "knapsack_demo.html")
    if os.path.exists(demo_path):
        with open(demo_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Demo not found</h1>")


@app.get("/{filename}")
async def serve_static(filename: str):
    filepath = os.path.join(STATIC_DIR, filename)
    if os.path.exists(filepath) and not os.path.isdir(filepath):
        content_type = "text/html"
        if filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".css"):
            content_type = "text/css"
        elif filename.endswith(".js"):
            content_type = "application/javascript"
        with open(filepath) as f:
            return HTMLResponse(f.read(), media_type=content_type)
    return JSONResponse({"error": "not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TunedAI Reasoning Engine — API Server")
    print("=" * 60)
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions          — Reasoning (T1-T4)")
    print(f"    POST /v1/chat/completions/baseline  — Generic (comparison)")
    print(f"    POST /api/agent                     — Simplified agent API")
    print(f"    GET  /health                        — Backend status")
    print(f"    GET  /agent                         — Interactive demo")
    print(f"    GET  /demo                          — Scripted demo")
    print("=" * 60)
    print(f"  Open: http://localhost:9001/agent")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=9001)
