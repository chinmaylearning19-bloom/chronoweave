"""
symbolic_guardrails.py — ChronoWeave Brain Layer
=================================================
Five symbolic guardrail rules that intercept proposed_order (or proposed_mrp)
before it executes in the physics engine.

Nodes
-----
  Node 1 : store              — Retail Stores
  Node 2 : central_warehouse  — Central Warehouse → Mill
  Node 3 : mill               — Fabric Mill
  Node 4 : procurement        — Procurement / Pricing

Each guardrail exposes a module-level `apply(...)` function and is also wired
into the master `SymbolicGuardrailShield` class.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Shared return type
# ---------------------------------------------------------------------------

GuardrailResult = dict  # keys: final_order, triggered, rule, trace, savings


# ===========================================================================
# Guardrail #1 — Return-Wave Netting
# Node 2: Central Warehouse → Mill
# ===========================================================================

def guardrail_1_return_wave_netting(
    proposed_order: int,
    physics_state: dict,
    stats_forecast: dict,
    macro: dict,
) -> GuardrailResult:
    """
    Deduct the predicted 10-day return wave from the factory order so that
    incoming returns fill demand before new production is triggered.

    Args:
        proposed_order  : units the replenishment model wants to order from mill.
        physics_state   : must contain "node" (str) and optionally
                          "predicted_returns_10day" (int ≥ 0).
        stats_forecast  : (unused by this rule, available for consistency).
        macro           : (unused by this rule, available for consistency).

    Returns:
        GuardrailResult dict.
    """
    node = physics_state.get("node", "")
    predicted_returns = int(physics_state.get("predicted_returns_10day", 0))

    triggered = node == "central_warehouse" and predicted_returns > 0

    if triggered:
        final_order = max(0, proposed_order - predicted_returns)
        netted = proposed_order - final_order
        trace = (
            f"Netted {netted} predicted returns from factory order. "
            "Prevents phantom reorder of ghost inventory."
        )
        savings = f"~{netted} units avoided; est. ₹{netted * 350:,} excess WIP prevented"
    else:
        final_order = proposed_order
        trace = "Return-Wave Netting: conditions not met (node or predicted_returns)."
        savings = "₹0"

    return {
        "final_order": final_order,
        "triggered": triggered,
        "rule": "GR1_RETURN_WAVE_NETTING",
        "trace": trace,
        "savings": savings,
    }


# ===========================================================================
# Guardrail #2 — Bullwhip Slew-Rate Ceiling
# Node 2: central_warehouse  (κ2 = 1.15)
# Node 3: mill               (κ3 = 1.10)
# ===========================================================================

# Per-node variance amplification cap
_KAPPA: dict[str, float] = {
    "central_warehouse": 1.15,
    "mill": 1.10,
}


def guardrail_2_bullwhip_slew_rate(
    proposed_order: int,
    physics_state: dict,
    stats_forecast: dict,
    macro: dict,
    order_history: Optional[deque] = None,
) -> GuardrailResult:
    """
    Clamp order variance so Var(u_i) / Var(D̂_organic) ≤ κ_i.

    Args:
        proposed_order  : units proposed by replenishment model.
        physics_state   : must contain "node" (str).
        stats_forecast  : must contain "organic_demand_history" (list[float],
                          last ≥2 weeks) and optionally "organic_demand" (float).
        macro           : (unused).
        order_history   : external deque(maxlen=8) maintained by the shield.
                          If None, a fresh deque is used (no history → no clamp).

    Returns:
        GuardrailResult dict.
    """
    node = physics_state.get("node", "")
    kappa = _KAPPA.get(node)

    if kappa is None:
        return {
            "final_order": proposed_order,
            "triggered": False,
            "rule": "GR2_BULLWHIP_SLEW_RATE",
            "trace": f"Bullwhip Slew-Rate: node '{node}' not subject to this guardrail.",
            "savings": "₹0",
        }

    # Compute Var(D̂_organic) from history
    organic_hist: list[float] = stats_forecast.get("organic_demand_history", [])
    if len(organic_hist) < 2:
        return {
            "final_order": proposed_order,
            "triggered": False,
            "rule": "GR2_BULLWHIP_SLEW_RATE",
            "trace": "Bullwhip Slew-Rate: insufficient demand history to compute variance.",
            "savings": "₹0",
        }

    var_demand = statistics.variance(organic_hist)
    organic_demand = stats_forecast.get("organic_demand", float(organic_hist[-1]))

    # Update order history
    if order_history is None:
        order_history = deque(maxlen=8)
    order_history.append(proposed_order)

    if len(order_history) < 2:
        return {
            "final_order": proposed_order,
            "triggered": False,
            "rule": "GR2_BULLWHIP_SLEW_RATE",
            "trace": "Bullwhip Slew-Rate: building order history — no clamp yet.",
            "savings": "₹0",
        }

    var_orders = statistics.variance(list(order_history))

    # Avoid division by zero
    if var_demand < 1e-9:
        ratio = 0.0
    else:
        ratio = var_orders / var_demand

    if ratio > kappa:
        # Clamp: order ≤ organic_demand * (1 + sqrt(κ - 1))
        ceil_multiplier = 1.0 + math.sqrt(kappa - 1.0)
        clamped_order = int(round(organic_demand * ceil_multiplier))
        final_order = min(proposed_order, max(0, clamped_order))
        triggered = True
        saved_units = proposed_order - final_order
        trace = (
            f"Slew-rate ceiling applied. Order variance ratio was {ratio:.2f}x "
            f"vs limit {kappa}x. Order clamped from {proposed_order} → {final_order}."
        )
        savings = f"~{saved_units} bullwhip units dampened; est. ₹{saved_units * 280:,} over-order prevented"
    else:
        final_order = proposed_order
        triggered = False
        trace = (
            f"Bullwhip Slew-Rate: variance ratio {ratio:.2f}x within limit {kappa}x. "
            "No clamp applied."
        )
        savings = "₹0"

    return {
        "final_order": final_order,
        "triggered": triggered,
        "rule": "GR2_BULLWHIP_SLEW_RATE",
        "trace": trace,
        "savings": savings,
        "_ratio": round(ratio, 4),   # internal debug field
        "_kappa": kappa,
    }


# ===========================================================================
# Guardrail #3 — Lateral Store Size Consolidation
# Node 1: Retail Stores
# ===========================================================================

def guardrail_3_lateral_size_consolidation(
    proposed_order: int,
    physics_state: dict,
    stats_forecast: dict,
    macro: dict,
) -> GuardrailResult:
    """
    If flagship stores have broken M/L size curves AND slow stores hold idle
    M/L stock, trigger a lateral transfer instead of a factory reorder.

    Args:
        proposed_order  : factory reorder units proposed.
        physics_state   : must contain:
            "node"                  : str  — should be "store"
            "flagship_ML_completeness" : float  — 0-1 rack completeness for M/L
            "slow_store_ML_excess"  : int   — excess M/L units in slow stores
        stats_forecast  : (unused).
        macro           : (unused).

    Returns:
        GuardrailResult dict.
    """
    node = physics_state.get("node", "")
    flagship_ml = float(physics_state.get("flagship_ML_completeness", 1.0))
    slow_excess = int(physics_state.get("slow_store_ML_excess", 0))

    triggered = (
        node == "store"
        and flagship_ml < 0.3
        and slow_excess > 50
    )

    if triggered:
        units_transferred = min(slow_excess, proposed_order)
        final_order = max(0, proposed_order - units_transferred)
        trace = (
            f"Lateral transfer triggered: {units_transferred} M/L units rerouted "
            "from slow stores. Factory reorder blocked."
        )
        savings = (
            f"~{units_transferred} units lateral-transferred; "
            f"est. ₹{units_transferred * 450:,} new production avoided"
        )
    else:
        final_order = proposed_order
        trace = (
            "Lateral Size Consolidation: conditions not met "
            f"(flagship_ML={flagship_ml:.2f}, slow_excess={slow_excess}). "
            "Normal reorder proceeds."
        )
        savings = "₹0"

    return {
        "final_order": final_order,
        "triggered": triggered,
        "rule": "GR3_LATERAL_SIZE_CONSOLIDATION",
        "trace": trace,
        "savings": savings,
    }


# ===========================================================================
# Guardrail #4 — Greige Postponement & EOSS Suicide Window
# Node 3: Fabric Mill
# ===========================================================================

# Total pipeline lead time: mill → store ≈ 52 days ≈ 7.4 weeks
_TOTAL_LEAD_TIME_WEEKS: float = 7.4
_SEASON_CLIFF_WEEKS: int = 8  # 7.4 + 0.6 buffer


def guardrail_4_eoss_suicide_window(
    proposed_order: int,
    physics_state: dict,
    stats_forecast: dict,
    macro: dict,
) -> GuardrailResult:
    """
    If (season_end_week - current_week) ≤ total_lead_time + 14-day buffer,
    zero out new fabric orders.  Only greige conversion is permitted.

    Args:
        proposed_order  : new fabric order proposed.
        physics_state   : must contain:
            "node"                   : str   — should be "mill"
            "season_weeks_remaining" : int   — weeks until season close
        stats_forecast  : (unused).
        macro           : (unused).

    Returns:
        GuardrailResult dict.
    """
    node = physics_state.get("node", "")
    weeks_remaining = int(physics_state.get("season_weeks_remaining", 99))

    triggered = node == "mill" and weeks_remaining <= _SEASON_CLIFF_WEEKS

    if triggered:
        final_order = 0
        trace = (
            f"EOSS Suicide Window active. {weeks_remaining} weeks to season end "
            f"< lead time ({_TOTAL_LEAD_TIME_WEEKS}w + buffer). "
            "New fabric orders zeroed. Convert greige only."
        )
        savings = (
            f"~{proposed_order} new fabric units blocked; "
            f"est. ₹{proposed_order * 120:,} greige deadstock prevented"
        )
    else:
        final_order = proposed_order
        trace = (
            f"EOSS Suicide Window: {weeks_remaining} weeks remain — above "
            f"{_SEASON_CLIFF_WEEKS}-week cliff. New fabric orders proceed."
        )
        savings = "₹0"

    return {
        "final_order": final_order,
        "triggered": triggered,
        "rule": "GR4_EOSS_SUICIDE_WINDOW",
        "trace": trace,
        "savings": savings,
        "_weeks_remaining": weeks_remaining,
        "_cliff_threshold": _SEASON_CLIFF_WEEKS,
    }


# ===========================================================================
# Guardrail #5 — Indian GST ₹999 Tax Cliff Shield
# Node 4: Procurement / Pricing
# ===========================================================================

_GST_CLIFF_LOW: int = 1001
_GST_CLIFF_HIGH: int = 1085
_GST_SAFE_MRP: int = 999


def guardrail_5_gst_cliff_shield(
    proposed_order: int,   # reinterpreted as proposed_mrp when node == "procurement"
    physics_state: dict,
    stats_forecast: dict,
    macro: dict,
) -> GuardrailResult:
    """
    Block MRP from landing in the ₹1001–₹1085 GST 12%-band cliff.
    If cost pressure pushes proposed MRP into this range, lock at ₹999
    and switch to hedged yarn lots.

    Args:
        proposed_order  : for Node 4 this carries the proposed MRP (₹).
        physics_state   : must contain:
            "node"         : str  — should be "procurement"
            "proposed_mrp" : int  — proposed retail price (overrides proposed_order
                             if present; otherwise proposed_order is used as MRP)
        stats_forecast  : (unused).
        macro           : (unused).

    Returns:
        GuardrailResult dict.  "final_order" field carries the resolved MRP.
    """
    node = physics_state.get("node", "")
    proposed_mrp = int(physics_state.get("proposed_mrp", proposed_order))

    triggered = (
        node == "procurement"
        and _GST_CLIFF_LOW <= proposed_mrp <= _GST_CLIFF_HIGH
    )

    if triggered:
        final_order = _GST_SAFE_MRP   # lock MRP at ₹999
        trace = (
            f"GST cliff intercepted. MRP ₹{proposed_mrp} enters 12% GST band "
            f"(₹{_GST_CLIFF_LOW}–₹{_GST_CLIFF_HIGH}). "
            f"Locked at ₹{_GST_SAFE_MRP}. Switched to hedged yarn procurement."
        )
        savings = (
            f"MRP held at ₹{_GST_SAFE_MRP} (vs ₹{proposed_mrp}). "
            "Consumer price competitiveness preserved; margin protected via yarn hedge."
        )
    else:
        final_order = proposed_mrp
        trace = (
            f"GST Cliff Shield: proposed MRP ₹{proposed_mrp} outside cliff band "
            f"(₹{_GST_CLIFF_LOW}–₹{_GST_CLIFF_HIGH}). No intervention."
        )
        savings = "₹0"

    return {
        "final_order": final_order,
        "triggered": triggered,
        "rule": "GR5_GST_CLIFF_SHIELD",
        "trace": trace,
        "savings": savings,
        "_proposed_mrp": proposed_mrp,
        "_locked_mrp": _GST_SAFE_MRP if triggered else proposed_mrp,
    }


# ===========================================================================
# Master Shield — applies all relevant guardrails for a given node
# ===========================================================================

class SymbolicGuardrailShield:
    """
    Orchestrates all five symbolic guardrails for the ChronoWeave supply chain.

    Usage
    -----
    shield = SymbolicGuardrailShield()
    result = shield.project_onto_physics(
        node="central_warehouse",
        proposed_order=1200,
        physics_state={...},
        stats_forecast={...},
        macro={...},
    )
    """

    def __init__(self) -> None:
        self.order_history: dict[str, deque] = {
            node: deque(maxlen=8)
            for node in ["store", "central_warehouse", "mill", "procurement"]
        }
        self.trace_log: list[dict] = []   # all symbolic decisions, in order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stamp(self, node: str, week: int, result: GuardrailResult) -> None:
        """Append a timestamped entry to the trace log."""
        self.trace_log.append({
            "node": node,
            "week": week,
            "rule": result["rule"],
            "triggered": result["triggered"],
            "final_order": result["final_order"],
            "trace": result["trace"],
            "savings": result["savings"],
        })

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def project_onto_physics(
        self,
        node: str,
        proposed_order: int,
        physics_state: dict,
        stats_forecast: dict,
        macro: dict,
        week: int = 0,
    ) -> dict:
        """
        Apply all relevant guardrails for *node* sequentially.  Each guardrail
        receives the output of the previous one as its proposed_order.

        Parameters
        ----------
        node            : one of "store", "central_warehouse", "mill", "procurement"
        proposed_order  : raw order quantity (or MRP for node==procurement)
        physics_state   : live physics/inventory state dict; must include "node"
        stats_forecast  : statistical forecast dict
        macro           : macroeconomic signal dict
        week            : simulation week number (for logging)

        Returns
        -------
        dict with keys:
            final_order         : int — post-guardrail order / MRP
            guardrails_triggered: list[str] — rule names that fired
            trace_log           : list[dict] — full symbolic trace this call
            mrp_decision        : dict — GST guardrail result (Node 4 only)
        """
        # Inject node into physics_state so each guardrail can self-identify
        ps = {**physics_state, "node": node}

        guardrails_triggered: list[str] = []
        call_trace: list[dict] = []
        mrp_decision: dict = {}
        current_order = proposed_order

        # ---- GR1: Return-Wave Netting (central_warehouse only) ----------
        if node == "central_warehouse":
            r1 = guardrail_1_return_wave_netting(current_order, ps, stats_forecast, macro)
            current_order = r1["final_order"]
            call_trace.append(r1)
            self._stamp(node, week, r1)
            if r1["triggered"]:
                guardrails_triggered.append(r1["rule"])

        # ---- GR2: Bullwhip Slew-Rate (central_warehouse + mill) ---------
        if node in ("central_warehouse", "mill"):
            r2 = guardrail_2_bullwhip_slew_rate(
                current_order, ps, stats_forecast, macro,
                order_history=self.order_history[node],
            )
            current_order = r2["final_order"]
            call_trace.append(r2)
            self._stamp(node, week, r2)
            if r2["triggered"]:
                guardrails_triggered.append(r2["rule"])

        # ---- GR3: Lateral Size Consolidation (store only) ---------------
        if node == "store":
            r3 = guardrail_3_lateral_size_consolidation(current_order, ps, stats_forecast, macro)
            current_order = r3["final_order"]
            call_trace.append(r3)
            self._stamp(node, week, r3)
            if r3["triggered"]:
                guardrails_triggered.append(r3["rule"])

        # ---- GR4: EOSS Suicide Window (mill only) ------------------------
        if node == "mill":
            r4 = guardrail_4_eoss_suicide_window(current_order, ps, stats_forecast, macro)
            current_order = r4["final_order"]
            call_trace.append(r4)
            self._stamp(node, week, r4)
            if r4["triggered"]:
                guardrails_triggered.append(r4["rule"])

        # ---- GR5: GST Cliff Shield (procurement only) --------------------
        if node == "procurement":
            r5 = guardrail_5_gst_cliff_shield(current_order, ps, stats_forecast, macro)
            current_order = r5["final_order"]
            mrp_decision = r5
            call_trace.append(r5)
            self._stamp(node, week, r5)
            if r5["triggered"]:
                guardrails_triggered.append(r5["rule"])

        return {
            "final_order": current_order,
            "guardrails_triggered": guardrails_triggered,
            "trace_log": call_trace,
            "mrp_decision": mrp_decision,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_full_trace(self) -> list[dict]:
        """Return the complete symbolic decision history."""
        return list(self.trace_log)

    def get_node_trace(self, node: str) -> list[dict]:
        """Return all trace entries for a specific node."""
        return [e for e in self.trace_log if e["node"] == node]

    def summary_stats(self) -> dict:
        """High-level counts of guardrail activations per rule."""
        from collections import Counter
        c: Counter = Counter()
        for e in self.trace_log:
            if e["triggered"]:
                c[e["rule"]] += 1
        return dict(c)
