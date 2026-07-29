import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn[standard]", "langgraph", "langchain-google-genai",
    "langchain-core", "sse-starlette", "httpx"])

import os, json, asyncio, uuid, httpx
from typing import TypedDict
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from langgraph.graph import StateGraph, START, END

# ── Config ──────────────────────────────────────────
API = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"
# Read from environment, never hardcode a key in source.
#   Windows PowerShell:  $env:GEMINI_API_KEY = "your-key-here"
#   macOS/Linux:         export GEMINI_API_KEY=your-key-here
KEY = "YOUR_API_KEY_HERE"  # os.environ.get("GEMINI_API_KEY")
Q = {}  # rid -> (asyncio.Queue, loop)

# ── LLM ─────────────────────────────────────────────
def llm(prompt, sys="", js=False):
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if sys:
        body["systemInstruction"] = {"parts": [{"text": sys}]}
    if js:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    r = httpx.post(f"{API}?key={KEY}", json=body, timeout=120)
    resp = r.json()
    if "error" in resp:
        raise Exception(resp["error"].get("message", str(resp["error"])))
    t = resp["candidates"][0]["content"]["parts"][0]["text"]
    if js:
        t = t.strip()
        if t.startswith("```"):
            t = "\n".join(t.split("\n")[1:]).rsplit("```", 1)[0]
        return json.loads(t)
    return t

def emit(rid, evt, d):
    if rid in Q:
        q, loop = Q[rid]
        loop.call_soon_threadsafe(q.put_nowait, (evt, d))

# ── Personas ────────────────────────────────────────
PERSONAS = [
    ("Analyst",    "You are a precise analytical expert. Prioritize accuracy and technical depth."),
    ("Pragmatist", "You are a practical problem-solver. Prioritize clarity and actionable steps."),
    ("Innovator",  "You are a creative thinker. Offer innovative and alternative approaches."),
    ("Educator",   "You are a patient teacher. Use examples and analogies to explain clearly."),
    ("Critic",     "You are a critical reviewer. Identify edge cases, flaws, and improvements."),
]

# ── State ───────────────────────────────────────────
class S(TypedDict):
    query: str
    rid: str
    classification: dict
    round1_outputs: list
    jury_round1: dict
    round2_outputs: list
    jury: dict
    final: str

# ── Graph Nodes ─────────────────────────────────────
def classify(s):
    r = llm(s["query"],
        'Classify this query. Return JSON: {"type":"direct" or "multi","count":2-5,"reasoning":"why"}.'
        ' "direct"=simple/creative/opinion/single-answer. "multi"=code/analysis/strategy that benefits from multiple perspectives.', True)
    emit(s["rid"], "classification", r)
    return {"classification": r}

def direct_node(s):
    r = llm(s["query"])
    emit(s["rid"], "direct", {"output": r})
    return {"final": r}

def agents_round1(s):
    """Task 2 equivalent: every agent generates independently, no visibility of peers."""
    n = min(int(s["classification"].get("count", 3)), 5)
    outs = []
    for i in range(n):
        nm, sys = PERSONAS[i % len(PERSONAS)]
        emit(s["rid"], "round1_agent_start", {"id": i, "name": nm})
        o = llm(s["query"], sys)
        outs.append({"id": i, "name": nm, "output": o})
        emit(s["rid"], "round1_agent_done", {"id": i, "name": nm, "output": o})
    return {"round1_outputs": outs}

def jury_round1_node(s):
    """Interim review only - identifies issues per agent, does NOT pick a winner yet."""
    emit(s["rid"], "jury_round1_start", {})
    txt = "\n\n".join(f"=== Agent {o['id']} ({o['name']}) ===\n{o['output']}" for o in s["round1_outputs"])
    r = llm(
        f"Query: {s['query']}\n\nRound 1 candidate outputs:\n{txt}",
        'You are a neutral reviewer preparing agents for a revision round. Do NOT pick a '
        'winner here. For each agent, note unsupported claims, gaps, or points another agent '
        'covered better. Return JSON: {"feedback":[{"id":0,"issues":"","what_to_consider":""}],'
        '"cross_agent_notes":"shared observations across all agents"}',
        True,
    )
    emit(s["rid"], "jury_round1_done", r)
    return {"jury_round1": r}

def agents_round2(s):
    """
    Task 3: the swap. Each agent sees every OTHER agent's round-1 output plus JULI/jury's
    round-1 feedback specific to them, then regenerates (revises) their own answer.
    """
    feedback_by_id = {f.get("id"): f for f in s["jury_round1"].get("feedback", [])}
    cross_notes = s["jury_round1"].get("cross_agent_notes", "")
    outs = []
    for o in s["round1_outputs"]:
        i = o["id"]
        nm, sys = PERSONAS[i % len(PERSONAS)]
        peers = "\n\n".join(
            f"=== Agent {p['id']} ({p['name']}) ===\n{p['output']}"
            for p in s["round1_outputs"] if p["id"] != i
        )
        fb = feedback_by_id.get(i, {})
        prompt = (
            f"Original query: {s['query']}\n\n"
            f"Your previous answer:\n{o['output']}\n\n"
            f"Other agents' answers:\n{peers}\n\n"
            f"Reviewer feedback on your answer: {fb.get('issues', '')} {fb.get('what_to_consider', '')}\n\n"
            f"Cross-agent notes: {cross_notes}\n\n"
            "Revise your answer. Keep what still holds, fix or drop what doesn't, and "
            "incorporate anything valuable from the other agents on its own merits. Do not "
            "just copy another agent's answer - defend or change each point yourself."
        )
        emit(s["rid"], "round2_agent_start", {"id": i, "name": nm})
        revised = llm(prompt, sys)
        outs.append({"id": i, "name": nm, "output": revised})
        emit(s["rid"], "round2_agent_done", {"id": i, "name": nm, "output": revised})
    return {"round2_outputs": outs}

def jury_final_node(s):
    """Final evaluation - scores the REVISED (round 2) outputs and picks the result."""
    emit(s["rid"], "jury_start", {})
    txt = "\n\n".join(f"=== Agent {o['id']} ({o['name']}) ===\n{o['output']}" for o in s["round2_outputs"])
    r = llm(
        f"Query: {s['query']}\n\nRevised candidate outputs (after cross-exchange):\n{txt}",
        "You are a jury evaluator scoring the REVISED answers, after agents saw each "
        'other\'s round-1 work. Score each agent 0-100 on: accuracy, completeness, clarity, '
        'relevance, usefulness. Return JSON: {"evaluations":[{"id":0,"name":"","scores":'
        '{"accuracy":0,"completeness":0,"clarity":0,"relevance":0,"usefulness":0},'
        '"overall":0,"strengths":"","weaknesses":""}],"best_id":0,"reasoning":""}',
        True,
    )
    best = next((o for o in s["round2_outputs"] if o["id"] == r.get("best_id", 0)), s["round2_outputs"][0])
    emit(s["rid"], "jury_done", r)
    return {"jury": r, "final": best["output"]}

def route(s):
    t = s["classification"].get("type", "direct")
    return t if t in ("direct", "multi") else "direct"

# ── LangGraph ───────────────────────────────────────
g = StateGraph(S)
g.add_node("classify", classify)
g.add_node("direct", direct_node)
g.add_node("agents_round1", agents_round1)
g.add_node("jury_round1", jury_round1_node)
g.add_node("agents_round2", agents_round2)
g.add_node("jury_final", jury_final_node)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", route, {"direct": "direct", "multi": "agents_round1"})
g.add_edge("agents_round1", "jury_round1")
g.add_edge("jury_round1", "agents_round2")
g.add_edge("agents_round2", "jury_final")
g.add_edge("jury_final", END)
g.add_edge("direct", END)
graph = g.compile()

# ── FastAPI + SSE ───────────────────────────────────
app = FastAPI()

@app.post("/api/query")
async def handle(req: Request):
    d = await req.json()
    rid = str(uuid.uuid4())
    q = asyncio.Queue()
    Q[rid] = (q, asyncio.get_event_loop())

    async def run():
        try:
            await asyncio.to_thread(graph.invoke, {
                "query": d["query"], "rid": rid,
                "classification": {}, "round1_outputs": [], "jury_round1": {},
                "round2_outputs": [], "jury": {}, "final": ""
            })
        except Exception as e:
            await q.put(("error", {"message": str(e)}))
        finally:
            await q.put(None)

    asyncio.create_task(run())

    async def stream():
        try:
            while (item := await q.get()) is not None:
                yield {"event": item[0], "data": json.dumps(item[1])}
        finally:
            Q.pop(rid, None)

    return EventSourceResponse(stream())

app.mount("/", StaticFiles(directory="public", html=True))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
