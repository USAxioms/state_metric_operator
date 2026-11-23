"""
Pharmaceutical Cognitive State Ledger — Chat Application Backend

Application layer only. phar.py is imported and left unchanged.

Execution path:
    HTTP/UI -> pharmaceutical MDM + R³ workflow -> Λ Total optimization
             -> Cognitive State Ledger export -> SHA-256 response

This application currently exposes a pharmaceutical workflow, not a
general-purpose scientific or conversational reasoning engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_phar_module() -> types.ModuleType:
    """Load phar.py without editing it; tolerate a Markdown code-fence wrapper."""
    source_path = HERE / "phar.py"
    source = source_path.read_text(encoding="utf-8")

    if "```python" in source:
        start = source.find("```python") + len("```python")
        end = source.rfind("```")
        if end <= start:
            raise ImportError(
                "phar.py has an opening Python fence but no closing fence"
            )
        source = source[start:end]

    module = types.ModuleType("phar")
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules["phar"] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


phar = _load_phar_module()

logging.getLogger().setLevel(logging.WARNING)

MAX_BODY_BYTES = 2_000_000
STATIC_DIR = HERE / "static"
STATIC_INDEX = STATIC_DIR / "index.html"

DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "safety_priority": "highest",
    "cost_target": 1000,
    "synthesis_green_chemistry": True,
}

# Explicit indicators that a request belongs to the pharmaceutical workflow.
# General science, chemistry, biology, energy, and materials questions do not
# qualify merely because they mention a molecule or substance.
PHARMA_INTENT_TERMS = re.compile(
    r"\b("
    r"drug discovery|drug candidate|drug development|pharmaceutical|"
    r"therapeutic|therapy|treatment|medication|medicine|"
    r"active pharmaceutical ingredient|api synthesis|"
    r"drug target|target engagement|"
    r"pharmacokinetic|pharmacodynamic|adme|"
    r"dose response|clinical trial|clinical evidence|"
    r"disease mechanism|disease target|"
    r"binding affinity|inhibitor|agonist|antagonist|"
    r"toxicity|therapeutic index|"
    r"formulation|bioavailability|"
    r"molecular scaffold|lead compound|"
    r"compound optimization|"
    r"indication"
    r")\b",
    re.IGNORECASE,
)

PHARMA_NEGATION = re.compile(
    r"\b(?:do not|don't|dont|not|never|without|exclude|avoid)\b"
    r".{0,80}\b(?:pharmaceutical|pharma|drug discovery|drug-development)\b",
    re.IGNORECASE | re.DOTALL,
)


def is_pharmaceutical_request(
    prompt: str,
    uploaded_target: dict[str, Any] | None,
) -> bool:
    """
    Route only clearly pharmaceutical requests into phar.py.

    A structured target counts as pharmaceutical only when the caller supplies
    an explicit disease/indication or a non-generic target class. A plain
    substance name or general scientific question is not enough.
    """
    data = uploaded_target if isinstance(uploaded_target, dict) else {}

    explicit_disease = bool(
        isinstance(data.get("disease"), str) and data["disease"].strip()
    )
    explicit_class = str(data.get("target_class") or "").strip().lower()
    explicit_pharma_class = explicit_class not in ("", "general_domain", "general")

    if explicit_disease or explicit_pharma_class:
        return True

    cleaned = " ".join((prompt or "").strip().split())
    if not cleaned or PHARMA_NEGATION.search(cleaned):
        return False

    return bool(PHARMA_INTENT_TERMS.search(cleaned))


def build_target_from_prompt(
    prompt: str,
    uploaded_target: dict[str, Any] | None,
) -> phar.TherapeuticTarget:
    """Build the existing phar.TherapeuticTarget input without changing phar.py."""
    data = uploaded_target if isinstance(uploaded_target, dict) else {}
    cleaned_prompt = " ".join((prompt or "").strip().split())

    raw_name = str(data.get("target_name") or data.get("target") or "").strip()
    if not raw_name:
        tokens = re.findall(r"[A-Za-z0-9_]+", cleaned_prompt)
        raw_name = "_".join(tokens[:4]) if tokens else "Domain_Inquiry"

    raw_disease = str(
        data.get("disease") or cleaned_prompt or "General Domain Inquiry"
    ).strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").upper() or "DOMAIN"

    return phar.TherapeuticTarget(
        target_id=str(data.get("target_id") or f"{slug}_001"),
        target_name=raw_name,
        target_class=str(data.get("target_class") or "general_domain"),
        disease=raw_disease,
        disease_prevalence=float(data.get("disease_prevalence", 100.0)),
        validation_score=float(data.get("validation_score", 0.85)),
        druggability_score=float(data.get("druggability_score", 0.75)),
        clinical_evidence=list(data.get("clinical_evidence") or ["User-provided prompt"]),
        known_modulators=list(data.get("known_modulators") or []),
        market_size_usd=float(data.get("market_size_usd", 1000.0)),
        unmet_need_score=float(data.get("unmet_need_score", 0.85)),
    )


def _build_invention_state(
    candidate: phar.DrugCandidate,
    pathway: phar.ReasoningPathway,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Translate the actual pharmaceutical output into Λ Total's input schema."""
    safety_ok = all(step.safety_validated for step in pathway.steps)
    return {
        "novel_mechanisms": [],
        "novelty_score": float(candidate.novelty_score),
        "novelty_level": (
            "breakthrough" if candidate.novelty_score >= 0.80 else "significant"
        ),
        "ethical_validation": {
            "ethical_score": 0.95 if safety_ok else 0.50,
            "patient_safety_primacy": safety_ok,
            "equitable_access": float(candidate.development_cost_estimate)
            <= float(constraints.get("cost_target", 1000)),
            "environmental_impact": bool(
                constraints.get("synthesis_green_chemistry", True)
            ),
        },
        "safety_profile": {
            "predicted_safety_index": float(candidate.predicted_safety_index),
            "pathway_steps_marked_safe": safety_ok,
        },
        "development_timeline_months": int(
            constraints.get("development_timeline_months", 60)
        ),
        "development_cost_estimate": float(candidate.development_cost_estimate),
    }


def _serialize_lambda_metrics(metrics: phar.LambdaMetrics) -> dict[str, Any]:
    """Return Λ Total metrics and recorded reflection data as JSON data."""
    return {
        "lambda_total": metrics.lambda_total,
        "u_substrate": metrics.u_substrate,
        "aiv_intelligent_value": metrics.aiv_intelligent_value,
        "alpha_decision_accuracy": metrics.alpha_decision_accuracy,
        "e_ethical_constraint": metrics.e_ethical_constraint,
        "r_resource_cost": metrics.r_resource_cost,
        "optimization_potential": metrics.optimization_potential,
        "improvement_rate": metrics.improvement_rate,
        "convergence_score": metrics.convergence_score,
        "optimization_iteration": metrics.optimization_iteration,
        "refinement_iteration": metrics.refinement_iteration,
        "validation_passed": metrics.validation_passed,
        "causality_trace": metrics.causality_trace,
        "bottleneck_analysis": metrics.bottleneck_analysis,
        "timestamp": metrics.timestamp,
    }


def format_natural_language_response(
    prompt: str,
    target: phar.TherapeuticTarget,
    candidate: phar.DrugCandidate,
    pathway: phar.ReasoningPathway,
    pharma_engine: phar.PharmaceuticalSuperintelligence,
    lambda_metrics: phar.LambdaMetrics,
    lambda_engine: phar.LambdaOptimizationEngine,
) -> str:
    total_axioms = len(phar.PharmaceuticalAxiomRegistry.get_foundational_axioms())
    axiom_coverage = len(pathway.axioms_used) / total_axioms if total_axioms else 0.0
    all_steps_safe = all(step.safety_validated for step in pathway.steps)
    criteria_status = (
        "satisfied"
        if pharma_engine.superintelligence_verified
        else "not fully satisfied"
    )

    lines = [
        f'In response to your prompt—"{prompt.strip()}"—the pharmaceutical workflow '
        f"processed target {target.target_name} ({target.target_id}).",
        "",
        "PHARMACEUTICAL REASONING AND REFINEMENT",
        f"Pathway {pathway.pathway_id} completed "
        f"{pharma_engine.r3_processor.iteration_count} R³ refinement cycles, "
        f"with {len(pathway.steps)} recorded steps and {len(pathway.axioms_used)} "
        f"of {total_axioms} foundational axioms represented "
        f"({axiom_coverage:.1%} coverage).",
        f"Pathway strength: {pathway.pathway_strength:.3f}; "
        f"mean step confidence: {pathway.total_confidence:.3f}; "
        f"emergence metric: {pathway.emergence_metric:.3f}.",
    ]

    for step in pathway.steps:
        lines.append(
            f"- {step.step_id} ({step.step_type.value}): {step.transformation} "
            f"| axioms: {', '.join(step.grounding_axioms)} "
            f"| MDM components: {', '.join(step.mdm_components_used)} "
            f"| confidence: {step.confidence:.2f} "
            f"| safety flag: {'true' if step.safety_validated else 'false'}"
        )

    lines.extend([
        "",
        "Λ TOTAL OPTIMIZATION",
        f"Λ Total score: {lambda_metrics.lambda_total:.4f}",
        f"U_Sub: {lambda_metrics.u_substrate:.4f} | "
        f"AIV: {lambda_metrics.aiv_intelligent_value:.4f} | "
        f"α_Dec: {lambda_metrics.alpha_decision_accuracy:.4f}",
        f"E_C: {lambda_metrics.e_ethical_constraint:.4f} | "
        f"R_C: {lambda_metrics.r_resource_cost:.4f}",
        f"Optimization potential: {lambda_metrics.optimization_potential:.1%}; "
        f"convergence score: {lambda_metrics.convergence_score:.4f}; "
        f"refinement cycles executed: {len(lambda_engine.refinement_cycles)}.",
        "",
        "SYNTHESIZED PHARMACEUTICAL OUTPUT RECORD",
        f"Compound record: {candidate.compound_id}",
        f"Predicted efficacy: {candidate.predicted_efficacy * 100:.1f}%",
        f"Predicted safety index: {candidate.predicted_safety_index:.2f}",
        f"Novelty score: {candidate.novelty_score * 100:.1f}%",
        f"Estimated development cost: USD "
        f"{candidate.development_cost_estimate:.1f} million.",
        "",
        "VERIFICATION QUALIFICATION",
        f"The pharmaceutical engine's internal criteria currently report: {criteria_status}.",
        f"All pathway step safety flags are true: {all_steps_safe}.",
        "These are implementation-level checks, not independent scientific validation, "
        "clinical evidence, or formal proof of safety. Candidate properties in phar.py "
        "include simulated/placeholder values. Λ Total metrics are calculated from "
        "the inputs and scoring functions implemented in phar.py; they are not external certification.",
    ])
    return "\n".join(lines)


def run_chat_turn(request_data: dict[str, Any]) -> dict[str, Any]:
    prompt = ""
    messages = request_data.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                prompt = msg["content"].strip()
                break

    if not prompt:
        for key in ("message", "prompt"):
            value = request_data.get(key)
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break

    if not prompt:
        raise ValueError("A non-empty user prompt is required.")

    uploaded_target = request_data.get("target")
    if not isinstance(uploaded_target, dict):
        uploaded_target = None

    # IMPORTANT: Do not manufacture a TherapeuticTarget from an arbitrary
    # question. phar.py exposes a pharmaceutical engine, not a general-purpose
    # science-answering engine. Non-pharmaceutical prompts stop here.
    if not is_pharmaceutical_request(prompt, uploaded_target):
        message = (
            "This application currently exposes its pharmaceutical MDM + R³ "
            "workflow and Λ Total optimization. Your request was not routed "
            "through that workflow because it does not clearly specify a "
            "pharmaceutical or therapeutic task. No TherapeuticTarget, drug "
            "candidate, efficacy estimate, safety index, or pharmaceutical "
            "ledger entry was generated. The current phar.py interface does "
            "not expose a general-purpose scientific answer engine, so this "
            "app cannot produce the requested general explanation without "
            "adding a separate general-domain engine."
        )
        return {
            "ok": True,
            "reply": message,
            "sha256": None,
            "summary": {
                "routed": False,
                "route": "unsupported_general_domain",
                "reason": "No pharmaceutical intent detected.",
            },
            "ledger": None,
        }

    constraints = dict(DEFAULT_CONSTRAINTS)
    if isinstance(request_data.get("constraints"), dict):
        constraints.update(request_data["constraints"])

    # 1. Construct a target only after pharmaceutical intent is established.
    target = build_target_from_prompt(prompt, uploaded_target)

    # 2. Run the existing pharmaceutical engine. Its discover method invokes
    #    the MDM pathway and its internal R³ refinement cycles.
    pharma_engine = phar.PharmaceuticalSuperintelligence()
    candidate = pharma_engine.discover_drug_candidate(target, constraints)

    if not pharma_engine.reasoning_pathways:
        raise RuntimeError("The pharmaceutical engine returned no reasoning pathway.")
    pathway = next(reversed(pharma_engine.reasoning_pathways.values()))

    # 3. Run the existing Λ Total optimizer against the actual candidate,
    #    pathway, and target. phar.py remains unchanged.
    lambda_engine = phar.LambdaOptimizationEngine()
    invention_state = _build_invention_state(candidate, pathway, constraints)
    lambda_metrics = lambda_engine.execute_refinement_cycle(
        invention_state=invention_state,
        pathway=pathway,
        target=target,
        max_cycles=int(constraints.get("lambda_max_cycles", 5)),
    )

    reply_text = format_natural_language_response(
        prompt,
        target,
        candidate,
        pathway,
        pharma_engine,
        lambda_metrics,
        lambda_engine,
    )

    ledger_export = pharma_engine.export_ledger_json()
    ledger_export["lambda_total_optimization"] = {
        "metrics": _serialize_lambda_metrics(lambda_metrics),
        "refinement_cycles": [
            {
                "cycle": item["cycle"],
                "suggestions": item["suggestions"],
                "metrics": _serialize_lambda_metrics(item["metrics"]),
            }
            for item in lambda_engine.refinement_cycles
        ],
        "converged": lambda_engine.converged,
        "best_lambda": lambda_engine.best_lambda,
    }

    canonical_bytes = json.dumps(
        ledger_export,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()

    return {
        "ok": True,
        "reply": reply_text,
        "sha256": digest,
        "summary": {
            "routed": True,
            "route": "pharmaceutical",
            "target_id": target.target_id,
            "target_name": target.target_name,
            "disease": target.disease,
            "compound_id": candidate.compound_id,
            "pathway_id": pathway.pathway_id,
            "pathway_strength": pathway.pathway_strength,
            "emergence_metric": pathway.emergence_metric,
            "total_confidence": pathway.total_confidence,
            "superintelligence_verified": pharma_engine.superintelligence_verified,
            "lambda_total": lambda_metrics.lambda_total,
            "lambda_converged": lambda_engine.converged,
            "lambda_refinement_cycles": len(lambda_engine.refinement_cycles),
        },
        "ledger": ledger_export,
    }


class LedgerHandler(BaseHTTPRequestHandler):
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
        request_path = urlparse(self.path).path
        if request_path in ("/", "/index.html"):
            if STATIC_INDEX.is_file():
                self._send_html(200, STATIC_INDEX.read_bytes())
            else:
                self._send_json(404, {
                    "ok": False,
                    "error": "static/index.html not found",
                    "app_location": str(HERE),
                    "static_location": str(STATIC_DIR),
                    "index_location": str(STATIC_INDEX),
                })
            return

        if request_path == "/api/status":
            self._send_json(200, {
                "ok": True,
                "engines": [
                    "phar.PharmaceuticalSuperintelligence (MDM + R³)",
                    "phar.LambdaOptimizationEngine (Λ Total)",
                ],
                "routing": "pharmaceutical-intent-gated",
                "general_domain_engine": False,
                "axioms_loaded": len(
                    phar.PharmaceuticalAxiomRegistry.get_foundational_axioms()
                ),
                "static_index_present": STATIC_INDEX.is_file(),
            })
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header"})
            return

        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {
                "ok": False,
                "error": "Request body missing or exceeds size limit",
            })
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

        try:
            result = run_chat_turn(parsed)
            self._send_json(200, result)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            logging.exception("Chat turn failed")
            self._send_json(500, {
                "ok": False,
                "error": "Cognitive execution failed.",
                "detail": str(exc),
            })

    def log_message(self, format_: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST") or (
        "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    )
    server = ThreadingHTTPServer((host, port), LedgerHandler)
    print("=" * 60, flush=True)
    print("PHARMACEUTICAL COGNITIVE STATE LEDGER — STARTUP", flush=True)
    print(f"APP LOCATION: {HERE}", flush=True)
    print(f"STATIC LOCATION: {STATIC_DIR}", flush=True)
    print(f"INDEX FILE EXISTS: {STATIC_INDEX.is_file()}", flush=True)
    print("ACTIVE EXECUTION: pharmaceutical-intent-gated MDM + R³, followed by Λ Total", flush=True)
    print(f"Listening on http://{host}:{port}", flush=True)
    print("=" * 60, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
