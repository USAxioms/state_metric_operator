"""
Pharmaceutical Cognitive State Ledger — Chat Application Backend (app.py)

Routes natural-language prompts directly into `phar.py` (`PharmaceuticalSuperintelligence`)
without fallback prompts or external APIs, serving `static/index.html` on `/` and
returning natural-language responses alongside the ledger output.

Run:
    python3 app.py

Environment variables:
    PORT    HTTP port (default: 8000)
    HOST    Bind address (default: 0.0.0.0 on Render, 127.0.0.1 locally)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phar  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

MAX_BODY_BYTES = 2_000_000
STATIC_INDEX = HERE / "static" / "index.html"

DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "safety_priority": "highest",
    "cost_target": 1000,
    "synthesis_green_chemistry": True,
}


def build_target_from_prompt(prompt: str, uploaded_target: dict[str, Any] | None) -> phar.TherapeuticTarget:
    """Build a target context directly from any prompt or JSON target without rejecting domains."""
    data = uploaded_target if isinstance(uploaded_target, dict) else {}
    cleaned_prompt = " ".join((prompt or "").strip().split())

    raw_name = str(data.get("target_name") or data.get("target") or "").strip()
    if not raw_name:
        tokens = re.findall(r"[A-Za-z0-9_]+", cleaned_prompt)
        raw_name = "_".join(tokens[:4]) if tokens else "Domain_Inquiry"

    raw_disease = str(data.get("disease") or cleaned_prompt or "General Domain Inquiry").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").upper() or "DOMAIN"

    return phar.TherapeuticTarget(
        target_id=str(data.get("target_id") or f"{slug}_001"),
        target_name=raw_name,
        target_class=str(data.get("target_class") or "general_domain"),
        disease=raw_disease,
        disease_prevalence=float(data.get("disease_prevalence", 100.0)),
        validation_score=float(data.get("validation_score", 0.85)),
        druggability_score=float(data.get("druggability_score", 0.75)),
        clinical_evidence=list(data.get("clinical_evidence") or ["Ledger reasoning trace"]),
        known_modulators=list(data.get("known_modulators") or ["Axiomatic and MDM synthesis"]),
        market_size_usd=float(data.get("market_size_usd", 1000.0)),
        unmet_need_score=float(data.get("unmet_need_score", 0.85)),
    )


def format_natural_language_response(
    prompt: str,
    target: phar.TherapeuticTarget,
    candidate: phar.DrugCandidate,
    pathway: phar.ReasoningPathway,
    engine: phar.PharmaceuticalSuperintelligence,
) -> str:
    """Render a natural-language response directly from the executed pathway and R3 state."""
    total_axioms = len(phar.PharmaceuticalAxiomRegistry.get_foundational_axioms())
    axiom_coverage = (len(pathway.axioms_used) / total_axioms * 100.0) if total_axioms else 0.0
    all_steps_safe = all(step.safety_validated for step in pathway.steps)
    criteria_status = "satisfied" if engine.superintelligence_verified else "not fully satisfied"

    latest_r3 = engine.r3_processor.metrics_history[-1] if engine.r3_processor.metrics_history else None
    r3_iterations = engine.r3_processor.iteration_count
    capability_score = latest_r3.capability_score if latest_r3 else engine.r3_processor.current_capability

    lines: list[str] = [
        f"In response to your prompt—\"{prompt.strip()}\"—the engine processed your input under context "
        f"{target.target_name} ({target.target_id}) through {r3_iterations} Recursive Refinement (R3) cycles.",
        "",
        "Reasoning and Refinement Output",
        f"Pathway {pathway.pathway_id} executed {len(pathway.steps)} steps across {len(pathway.axioms_used)} of "
        f"{total_axioms} foundational axioms ({axiom_coverage:.0f}% coverage), reaching a pathway strength of "
        f"{pathway.pathway_strength:.3f}, a mean step confidence of {pathway.total_confidence:.3f}, an "
        f"emergence metric (Em) of {pathway.emergence_metric:.3f}, and an R3 capability index of "
        f"{capability_score:.3f}.",
    ]

    for step in pathway.steps:
        axioms_list = ", ".join(step.grounding_axioms)
        mdm_list = ", ".join(step.mdm_components_used)
        safety_state = "validated" if step.safety_validated else "unvalidated"
        lines.append(
            f"- {step.step_id} ({step.step_type.value}): {step.transformation}. "
            f"Axioms: [{axioms_list}]; MDM: [{mdm_list}]; confidence: {step.confidence:.2f}; "
            f"safety: {safety_state}."
        )

    lines.extend([
        "",
        "Synthesized Output Record",
        f"The engine produced output record {candidate.compound_id} with predicted efficacy of "
        f"{candidate.predicted_efficacy * 100:.1f}%, safety index of {candidate.predicted_safety_index:.1f}, "
        f"novelty score of {candidate.novelty_score * 100:.1f}%, and estimated development cost of "
        f"USD {candidate.development_cost_estimate:.1f} million.",
        "",
        "Verification Status",
        f"Internal criteria checks (Em >= 2.0, step safety validation = {all_steps_safe}, axiom coverage "
        f">= 0.95) report overall verification as {criteria_status}. Certain candidate fields and safety "
        f"checks in `phar.py` use fixed or placeholder values inside the engine rather than external data or "
        f"formal external proofs.",
    ])

    return "\n".join(lines)


def run_chat_turn(request_data: dict[str, Any]) -> dict[str, Any]:
    """Run a single JSON chat request directly through the engine."""
    messages = request_data.get("messages")
    prompt = ""
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                prompt = msg["content"]
                break
    if not prompt and isinstance(request_data.get("message"), str):
        prompt = request_data["message"]
    if not prompt and isinstance(request_data.get("prompt"), str):
        prompt = request_data["prompt"]

    uploaded_target = request_data.get("target") if isinstance(request_data.get("target"), dict) else None
    constraints = {**DEFAULT_CONSTRAINTS}
    if isinstance(request_data.get("constraints"), dict):
        constraints.update(request_data["constraints"])

    target = build_target_from_prompt(prompt, uploaded_target)
    engine = phar.PharmaceuticalSuperintelligence()
    candidate = engine.discover_drug_candidate(target, constraints)
    pathway = list(engine.reasoning_pathways.values())[-1]
    reply_text = format_natural_language_response(prompt, target, candidate, pathway, engine)

    ledger_export = engine.export_ledger_json()
    canonical_bytes = json.dumps(ledger_export, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()

    return {
        "ok": True,
        "reply": reply_text,
        "sha256": digest,
        "summary": {
            "target_id": target.target_id,
            "target_name": target.target_name,
            "disease": target.disease,
            "compound_id": candidate.compound_id,
            "pathway_id": pathway.pathway_id,
            "pathway_strength": pathway.pathway_strength,
            "emergence_metric": pathway.emergence_metric,
            "total_confidence": pathway.total_confidence,
            "superintelligence_verified": engine.superintelligence_verified,
        },
        "ledger": ledger_export,
    }


class LedgerHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving `static/index.html`, `/api/status`, and `/api/chat`."""

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, status_code: int, html_bytes: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(html_bytes)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if STATIC_INDEX.is_file():
                self._send_html(200, STATIC_INDEX.read_bytes())
            else:
                self._send_json(404, {"ok": False, "error": "static/index.html not found"})
        elif self.path == "/api/status":
            self._send_json(200, {
                "ok": True,
                "engine": "phar.PharmaceuticalSuperintelligence",
                "axioms_loaded": len(phar.PharmaceuticalAxiomRegistry.get_foundational_axioms()),
            })
        else:
            self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        content_length_raw = self.headers.get("Content-Length") or "0"
        try:
            length = int(content_length_raw)
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header"})
            return

        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"ok": False, "error": "Request body missing or exceeds size limit"})
            return

        body_bytes = self.rfile.read(length)
        try:
            parsed = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "Request body must be valid JSON"})
            return

        if not isinstance(parsed, dict):
            self._send_json(400, {"ok": False, "error": "JSON payload must be an object"})
            return

        result = run_chat_turn(parsed)
        self._send_json(200, result)

    def log_message(self, format_: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1")
    server = ThreadingHTTPServer((host, port), LedgerHandler)
    print(f"Cognitive State Ledger chat server listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
