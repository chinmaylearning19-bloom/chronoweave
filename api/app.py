"""
app.py — ChronoWeave Flask API Server
Indian Omnichannel Apparel Supply Chain Simulator & Bullwhip Mitigator.
"""

import os
import sys
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

# Ensure paths
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_root, 'brain'))
sys.path.insert(0, os.path.join(_root, 'data'))
sys.path.insert(0, _here)

load_dotenv()

from simulation_engine import ChronoWeaveEngine

static_dir = os.path.join(_root, "frontend")
app = Flask(__name__, static_folder=static_dir, static_url_path="")
engine = ChronoWeaveEngine()


# ── Frontend Route ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Simulation Routes ──────────────────────────────────────────────────────────

@app.route("/api/step", methods=["POST"])
def step():
    """Advance simulation by 1 week."""
    try:
        payload = engine.step()
        return jsonify({"ok": True, "data": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def run_bulk():
    """Run N weeks at once (default 10)."""
    try:
        body = request.get_json(force=True) or {}
        weeks = min(int(body.get("weeks", 10)), 104)
        results = engine.run_bulk(weeks)
        return jsonify({"ok": True, "weeks_run": len(results), "data": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/state", methods=["GET"])
def get_state():
    """Get latest week's state."""
    if engine.week == 0:
        return jsonify({"ok": True, "week": 0, "message": "Simulation not started"})
    latest = engine.sim_log[-1] if engine.sim_log else {}
    return jsonify({
        "ok": True,
        "week": engine.week,
        "latest": latest,
        "greige_batches": engine.greige_stock_batches,
        "season_weeks_remaining": engine.current_season_end - engine.week
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset simulation to Week 0."""
    engine.reset()
    return jsonify({"ok": True, "message": "ChronoWeave simulation reset to week 0"})


# ── Alert Routes ───────────────────────────────────────────────────────────────

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Get active alerts, filterable by ?node=... and ?category=..."""
    node = request.args.get("node")
    if engine.week == 0:
        return jsonify({"ok": True, "alerts": [], "summary": {}})

    alerts = engine.alert_engine.active_alerts
    if node:
        alerts = [a for a in alerts if a.node == node]

    alerts_json = [
        {
            "id": a.id, "node": a.node, "severity": a.severity.value,
            "category": a.category.value, "title": a.title, "detail": a.detail,
            "action": a.action, "metric_name": a.metric_name, "metric_value": a.metric_value,
            "threshold": a.threshold, "week": a.week, "auto_resolved": a.auto_resolved
        }
        for a in alerts
    ]
    summary = engine.alert_engine.get_dashboard_summary()
    return jsonify({"ok": True, "week": engine.week, "alerts": alerts_json, "summary": summary})


@app.route("/api/alerts/dismiss", methods=["POST"])
def dismiss_alert():
    """Dismiss an active alert."""
    body = request.get_json(force=True) or {}
    alert_id = body.get("id")
    if alert_id:
        engine.alert_engine.dismiss(alert_id)
        return jsonify({"ok": True, "dismissed": alert_id})
    return jsonify({"ok": False, "error": "Missing alert id"}), 400


# ── Information Bus & Cross-Node Visibility ───────────────────────────────────

@app.route("/api/cross_node/<node>", methods=["GET"])
def get_cross_node(node):
    """Retrieve full cross-node visibility digest from Information Bus for a node."""
    if engine.week == 0:
        return jsonify({"ok": True, "digest": {}})
    visibility = engine.info_bus.get_cross_node_visibility(node)
    return jsonify({"ok": True, "node": node, "visibility": visibility})


@app.route("/api/cascade_risks", methods=["GET"])
def get_cascade_risks():
    """Multi-node cascade risk detector from Information Bus."""
    if engine.week == 0:
        return jsonify({"ok": True, "risks": []})
    risks = engine.info_bus.detect_cascade_risk()
    return jsonify({"ok": True, "week": engine.week, "risks": risks})


# ── Chat Route (Node Specific Copilot) ─────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    LLM prompt query for specific supply chain node operator:
    'store', 'central_warehouse', 'mill', 'procurement'
    """
    body = request.get_json(force=True) or {}
    node = body.get("node", "store")
    message = body.get("message", "").strip()

    valid_nodes = ["store", "central_warehouse", "mill", "procurement"]
    if node not in valid_nodes:
        return jsonify({"ok": False, "error": f"Invalid node '{node}'. Valid: {valid_nodes}"}), 400
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    chat_result = engine.chat(node, message)
    return jsonify({
        "ok": True,
        "node": node,
        "week": engine.week,
        "response": chat_result.get("response", ""),
        "mode": chat_result.get("mode", "UNKNOWN"),
        "fallback": chat_result.get("fallback", False)
    })


@app.route("/api/chat/history/<node>", methods=["GET"])
def get_chat_history(node):
    """Get chat history for a node."""
    if not engine._chat:
        return jsonify({"ok": True, "history": []})
    history = engine._chat.get_history(node)
    return jsonify({"ok": True, "node": node, "history": history})


# ── Counterfactual Lookahead ("What-If") ───────────────────────────────────────

@app.route("/api/what_if", methods=["POST"])
def what_if():
    """
    Counterfactual lookahead: what if the operator changes this week's order?
    Simulates 2-week forward impact on working capital, deadstock, and stockout risk.
    """
    body = request.get_json(force=True) or {}
    node = body.get("node", "central_warehouse")
    delta_qty = int(body.get("delta_qty", 0))

    if engine.week == 0 or not engine.sim_log:
        return jsonify({"ok": False, "error": "Run simulation at least 1 week first"}), 400

    last_step = engine.sim_log[-1]
    base_order = last_step["final_orders"].get(node, 2000)
    adjusted_order = max(0, base_order + delta_qty)

    # Simplified forward model (2-week lookahead)
    unit_cost = 450.0 if node == "central_warehouse" else 650.0
    holding_rate = 0.005  # 0.5% per week
    wc_delta = delta_qty * unit_cost
    holding_delta_2w = wc_delta * holding_rate * 2

    # Stockout risk: if negative delta exceeds current inventory
    curr_inv = last_step["inventory"].get(node, 10000)
    stockout_prob_pct = max(0, min(100, int((abs(delta_qty) / max(curr_inv, 1)) * 100))) if delta_qty < 0 else 2

    # Deadstock risk: if positive delta near season end
    season_rem = engine.current_season_end - engine.week
    deadstock_risk_pct = min(100, int((max(0, 8 - season_rem) / 8) * 100)) if delta_qty > 0 else 5

    return jsonify({
        "ok": True,
        "node": node,
        "base_order": base_order,
        "adjusted_order": adjusted_order,
        "delta_qty": delta_qty,
        "working_capital_impact_inr": wc_delta,
        "holding_cost_impact_2w_inr": holding_delta_2w,
        "stockout_probability_pct": stockout_prob_pct,
        "deadstock_risk_pct": deadstock_risk_pct,
        "verdict": (
            "DANGEROUS: Order cut creates high risk of store stockout in core sizes."
            if stockout_prob_pct > 40 else
            "HIGH RISK: Ordering surplus near season end increases EOSS deadstock."
            if deadstock_risk_pct > 50 else
            "STABLE: Order adjustment within safe operational envelope."
        )
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5051))
    print(f"\n🚀 ChronoWeave API running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
