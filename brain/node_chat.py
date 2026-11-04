"""
node_chat.py — ChronoWeave LLM Node Query System
Physics-gated, role-specific conversational AI at every supply chain node.

Four distinct operator personas:
  Node 1 — Store Manager    : Size replenishment, stockout decisions, promo management
  Node 2 — Warehouse Manager: Return-wave netting, factory orders, fill-rate
  Node 3 — Mill Manager     : Greige conversion, season-end decisions, dye batches
  Node 4 — Procurement Mgr  : Yarn hedging, cotton index, GST cliff management

Each LLM call receives the full cross-node visibility snapshot from the InformationBus,
the current Wardenclyffe physics telemetry, all active alerts, and the guardrail trace log.
"""

import os
import json
import warnings
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

# SDK detection
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = os.getenv("MODEL_NAME", "gemini-2.0-flash")
GENAI_AVAILABLE = False
GENAI_SDK = "none"

try:
    from google import genai as _genai
    from google.genai import types as _gtypes
    if GEMINI_API_KEY:
        _client = _genai.Client(api_key=GEMINI_API_KEY)
    GENAI_AVAILABLE = bool(GEMINI_API_KEY)
    GENAI_SDK = "new"
except ImportError:
    try:
        import google.generativeai as _genai_legacy
        if GEMINI_API_KEY:
            _genai_legacy.configure(api_key=GEMINI_API_KEY)
        GENAI_AVAILABLE = bool(GEMINI_API_KEY)
        GENAI_SDK = "legacy"
    except ImportError:
        pass


# ── Node Personas ──────────────────────────────────────────────────────────────

NODE_PERSONAS = {
    "store": {
        "name":         "Store Operations Manager",
        "role_context": "You manage 20 omnichannel retail stores in India for an apparel brand. You handle size replenishment, visual merchandising, promotional events, and stockout decisions.",
        "primary_fears": ["Empty M/L rack during peak hours", "Promo overstock", "Wrong size mix delivered"],
        "primary_goals": ["Full-price sell-through", "Zero stockout on core sizes", "Minimum EOSS markdown"],
        "decision_horizon": "Daily/Weekly",
        "key_metrics": ["rack_completeness", "size_curve_health", "fill_rate", "pos_velocity"],
    },
    "central_warehouse": {
        "name":         "Central Warehouse Manager",
        "role_context": "You manage the central distribution hub serving 20 stores and ordering from the fabric mill. Your critical challenge is separating real demand from phantom orders (promo spikes + return boomerangs).",
        "primary_fears": ["Phantom reorder on promotional demand", "Return wave overwhelming intake capacity", "Overstocking before EOSS"],
        "primary_goals": ["Accurate fill-rate to stores", "Factory order grounded in organic demand", "Working capital efficiency"],
        "decision_horizon": "Weekly",
        "key_metrics": ["effective_inventory", "return_wave_forecast", "phantom_demand_split", "fill_rate", "factory_order_signal"],
    },
    "mill": {
        "name":         "Fabric Mill Manager",
        "role_context": "You manage the cut-make-trim facility and greige fabric conversion. Your most critical decision is whether to convert existing undyed greige or accept new fabric orders given season-end proximity.",
        "primary_fears": ["Committing fabric too close to season end", "Dye-vat idle capacity", "Emergency conversion with wrong colour"],
        "primary_goals": ["Smooth production schedule", "Zero new-fabric waste post-EOSS", "Timely greige conversion"],
        "decision_horizon": "2-4 Weeks",
        "key_metrics": ["greige_stock", "season_weeks_remaining", "production_queue", "lead_time_estimate", "dye_batch_status"],
    },
    "procurement": {
        "name":         "Procurement & Costing Manager",
        "role_context": "You manage yarn and raw material procurement, GST-sensitive pricing decisions, and hedging strategy. The ₹999 MRP threshold is your most critical operational constraint.",
        "primary_fears": ["MRP drifting into ₹1001-₹1085 GST cliff", "Spot yarn during cotton spike", "Over-hedging on falling commodity"],
        "primary_goals": ["Keep MRP at ₹999 or jump cleanly to ₹1199+", "Hedged yarn ≥ 60% of procurement", "Protect gross margin ≥ 55%"],
        "decision_horizon": "4-8 Weeks",
        "key_metrics": ["cotton_price_signal", "cost_pressure_index", "gst_cliff_status", "yarn_hedge_status", "mrp_decision"],
    },
}

CROSS_NODE_CONTEXT_KEYS = {
    "store":              ["central_warehouse", "macro"],
    "central_warehouse":  ["store", "mill", "macro"],
    "mill":               ["central_warehouse", "procurement", "macro"],
    "procurement":        ["mill", "macro"],
}


def _build_system_prompt(node: str, node_state: dict, cross_node_visibility: dict,
                          physics: dict, active_alerts: list, guardrail_log: list,
                          week: int) -> str:
    persona = NODE_PERSONAS.get(node, NODE_PERSONAS["store"])
    alerts_relevant = [a for a in active_alerts if a.get("node") == node or a.get("severity") == "CRITICAL"][:5]

    guardrails_fired = [g for g in guardrail_log if g.get("node") == node and g.get("triggered")]

    cross_ctx = {}
    for ctx_node in CROSS_NODE_CONTEXT_KEYS.get(node, []):
        if ctx_node in cross_node_visibility:
            cross_ctx[ctx_node] = cross_node_visibility[ctx_node]

    prompt = f"""You are ChronoWeave AI — an embedded supply chain intelligence copilot for a neurosymbolic omnichannel apparel simulation.

You are speaking directly to the **{persona['name']}** at **{node.upper().replace('_', ' ')}** — Week {week} of the simulation.

=== ROLE CONTEXT ===
{persona['role_context']}
Primary fears: {', '.join(persona['primary_fears'])}
Primary goals: {', '.join(persona['primary_goals'])}
Decision horizon: {persona['decision_horizon']}

=== YOUR NODE STATE (Week {week}) ===
{json.dumps({k: v for k, v in node_state.items() if not k.startswith('_')}, indent=2, default=str)}

=== WARDENCLYFFE PHYSICS TELEMETRY ===
- System Stress Energy (E_n): {physics.get('epistemic_energy', 'N/A')}
- Adaptive Learning Rate (η_s): {physics.get('eta_s', 'N/A')}
- Inference Mode: {physics.get('inference_mode', 'N/A')}
- Snapshot: #{physics.get('snapshot', 0)}
- Stress breakdown: {json.dumps(physics.get('stress_components', {}), default=str)}

=== CROSS-NODE VISIBILITY (from Information Bus) ===
{json.dumps(cross_ctx, indent=2, default=str)}

=== ACTIVE ALERTS FOR YOUR NODE ===
{json.dumps(alerts_relevant, indent=2, default=str) if alerts_relevant else "No active alerts."}

=== GUARDRAIL DECISIONS THIS WEEK ===
{json.dumps(guardrails_fired, indent=2, default=str) if guardrails_fired else "No guardrails triggered."}

=== INSTRUCTIONS ===
- Answer the operator's question directly and concisely — maximum 220 words.
- Ground every recommendation in the ACTUAL numbers from the state above (inventory units, return counts, ₹ cost, weeks remaining).
- If a symbolic guardrail has already auto-resolved the issue, explain what it did and confirm whether human override is needed.
- If the Wardenclyffe inference mode is CRISIS_RESPONSE or SYSTEMIC_LOCK, flag this prominently.
- For cross-node impacts (e.g., a promo at the store affecting the mill's production queue), explicitly call out the chain effect.
- Give a clear YES / NO / WAIT verdict first, then the rationale.
- Never give generic advice. Every statement must reference a specific number from the state above."""

    return prompt


def query_node(
    node: str,
    user_message: str,
    node_state: dict,
    cross_node_visibility: dict,
    physics: dict,
    active_alerts: list,
    guardrail_log: list,
    chat_history: list,
    week: int,
) -> dict:
    """
    Main entry point for node LLM queries.
    Returns: {response: str, node: str, week: int, mode: str, fallback: bool}
    """
    system_prompt = _build_system_prompt(
        node, node_state, cross_node_visibility,
        physics, active_alerts, guardrail_log, week
    )

    if not GENAI_AVAILABLE:
        return {
            "response": _deterministic_fallback(node, node_state, physics, active_alerts, guardrail_log, week),
            "node": node,
            "week": week,
            "mode": "FALLBACK",
            "fallback": True,
        }

    try:
        # Build history in correct format
        history = []
        for h in chat_history[-8:]:
            history.append({"role": h["role"], "parts": [{"text": h["content"]}]})
        history.append({"role": "user", "parts": [{"text": user_message}]})

        if GENAI_SDK == "new":
            resp = _client.models.generate_content(
                model=MODEL_NAME,
                contents=history,
                config=_gtypes.GenerateContentConfig(system_instruction=system_prompt),
            )
            text = resp.text.strip()
        else:
            model = _genai_legacy.GenerativeModel(MODEL_NAME, system_instruction=system_prompt)
            chat  = model.start_chat(history=[
                {"role": h["role"], "parts": [h["content"]]} for h in chat_history[-8:]
            ])
            text  = chat.send_message(user_message).text.strip()

        return {"response": text, "node": node, "week": week, "mode": "LLM", "fallback": False}

    except Exception as e:
        return {
            "response": _deterministic_fallback(node, node_state, physics, active_alerts, guardrail_log, week),
            "node": node,
            "week": week,
            "mode": "FALLBACK_ERROR",
            "fallback": True,
            "error": str(e),
        }


def _deterministic_fallback(node, node_state, physics, active_alerts, guardrail_log, week) -> str:
    """Structured deterministic response when LLM is unavailable."""
    persona    = NODE_PERSONAS.get(node, {})
    mode       = physics.get("inference_mode", "UNKNOWN")
    En         = physics.get("epistemic_energy", "N/A")
    eta_s      = physics.get("eta_s", "N/A")
    fired      = [g["rule"] for g in guardrail_log if g.get("triggered") and g.get("node") == node]
    alert_count = len([a for a in active_alerts if a.get("node") == node])

    lines = [
        f"[{persona.get('name', node.upper())} — Week {week}]",
        f"Wardenclyffe Mode: {mode} | E_n={En} | η_s={eta_s}",
        "",
    ]

    # Node-specific key metrics
    if node == "store":
        lines.append(f"Rack Completeness: {node_state.get('rack_completeness', 'N/A')}")
        lines.append(f"M/L Completeness: {node_state.get('ml_size_completeness', 'N/A')}")
        lines.append(f"POS Velocity: {node_state.get('pos_velocity', 'N/A')} units/day")
    elif node == "central_warehouse":
        lines.append(f"Effective Inventory: {node_state.get('effective_inventory', 'N/A')} units")
        lines.append(f"Predicted Returns (7-14d): {node_state.get('predicted_returns_week', 'N/A')} units")
        lines.append(f"Factory Order Signal: {node_state.get('factory_order', 'N/A')} units")
    elif node == "mill":
        lines.append(f"Greige Stock: {node_state.get('greige_stock_batches', 'N/A')} batches")
        lines.append(f"Season Weeks Remaining: {node_state.get('season_weeks_remaining', 'N/A')}")
        lines.append(f"Lead Time Estimate: {node_state.get('lead_time_estimate_days', 'N/A')} days")
    elif node == "procurement":
        lines.append(f"Cotton Index: {node_state.get('cotton_index', 'N/A')} ₹/kg")
        lines.append(f"Cost Pressure Index: {node_state.get('cost_pressure_index', 'N/A')}")
        lines.append(f"MRP Status: {node_state.get('mrp_status', 'N/A')}")

    if fired:
        lines.append(f"\nGuardrails Fired: {', '.join(fired)}")
    if alert_count:
        lines.append(f"Active Alerts: {alert_count}")

    return "\n".join(lines)


# ── Chat Session Manager ────────────────────────────────────────────────────────

class NodeChatSession:
    """Maintains per-node conversation history."""

    def __init__(self):
        self.histories: dict[str, list] = {
            n: [] for n in ["store", "central_warehouse", "mill", "procurement"]
        }

    def send(self, node: str, message: str, node_state: dict, cross_node_visibility: dict,
             physics: dict, active_alerts: list, guardrail_log: list, week: int) -> dict:
        history = self.histories.get(node, [])
        result  = query_node(
            node=node,
            user_message=message,
            node_state=node_state,
            cross_node_visibility=cross_node_visibility,
            physics=physics,
            active_alerts=active_alerts,
            guardrail_log=guardrail_log,
            chat_history=history,
            week=week,
        )
        history.append({"role": "user",  "content": message})
        history.append({"role": "model", "content": result["response"]})
        # Keep last 20 turns
        self.histories[node] = history[-20:]
        return result

    def clear(self, node: Optional[str] = None):
        if node:
            self.histories[node] = []
        else:
            for n in self.histories:
                self.histories[n] = []

    def get_history(self, node: str) -> list:
        return self.histories.get(node, [])
