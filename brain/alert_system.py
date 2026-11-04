"""
alert_system.py — ChronoWeave Brain Layer
==========================================
Typed, categorised, deduplicated alert engine.  Monitors all node states and
the information bus, generates structured alerts with specific recommended
actions, and surfaces them to the operator dashboard.

Severity tiers
--------------
  CRITICAL  — Immediate action required (EOSS cliff, GST breach, intake overload)
  HIGH      — Action required this week
  MEDIUM    — Monitor and prepare response
  LOW       — Informational

Categories
----------
  DEMAND · RETURNS · INVENTORY · GUARDRAIL · MACRO
  SEASONAL · FINANCIAL · OPERATIONAL
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    # For sorting (lower int = more severe)
    def __lt__(self, other: "AlertSeverity") -> bool:
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        return order[self.value] < order[other.value]


class AlertCategory(Enum):
    DEMAND = "DEMAND"
    RETURNS = "RETURNS"
    INVENTORY = "INVENTORY"
    GUARDRAIL = "GUARDRAIL"
    MACRO = "MACRO"
    SEASONAL = "SEASONAL"
    FINANCIAL = "FINANCIAL"
    OPERATIONAL = "OPERATIONAL"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """
    A single, fully-typed alert event.

    Fields
    ------
    id                  : unique alert identifier
    node                : originating supply chain node
    severity            : AlertSeverity enum value
    category            : AlertCategory enum value
    title               : short one-line summary
    detail              : full diagnostic explanation
    action              : specific recommended operator action
    metric_name         : name of the KPI that was breached
    metric_value        : observed value
    threshold           : threshold that was crossed
    week                : simulation week when alert was raised
    guardrail_triggered : guardrail rule ID (if category == GUARDRAIL)
    auto_resolved       : True if a guardrail already handled it automatically
    """
    id: str
    node: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    detail: str
    action: str
    metric_name: str
    metric_value: float
    threshold: float
    week: int
    guardrail_triggered: Optional[str] = None
    auto_resolved: bool = False

    # Deduplication key: same category + node suppresses within 2 weeks
    @property
    def dedup_key(self) -> str:
        return f"{self.category.value}|{self.node}|{self.metric_name}"


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Monitors all node states and the information bus.
    Generates structured alerts with specific recommended actions.
    Deduplicates and prioritises across nodes.

    Usage
    -----
    engine = AlertEngine(info_bus)
    alerts = engine.evaluate_week(week=12, all_node_states={...}, guardrail_log=[...])
    """

    # Suppress duplicate alert if same dedup_key raised within this many weeks
    _DEDUP_WINDOW_WEEKS: int = 2

    def __init__(self, info_bus) -> None:
        self.info_bus = info_bus
        self.active_alerts: list[Alert] = []
        self.alert_history: deque = deque(maxlen=52 * 10)   # ≈10 simulation years
        self._alert_counter: int = 0

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _new_id(self) -> str:
        self._alert_counter += 1
        return f"ALT-{self._alert_counter:05d}-{uuid.uuid4().hex[:6].upper()}"

    def _make(
        self,
        node: str,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        detail: str,
        action: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        week: int,
        guardrail_triggered: Optional[str] = None,
        auto_resolved: bool = False,
    ) -> Alert:
        return Alert(
            id=self._new_id(),
            node=node,
            severity=severity,
            category=category,
            title=title,
            detail=detail,
            action=action,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            week=week,
            guardrail_triggered=guardrail_triggered,
            auto_resolved=auto_resolved,
        )

    # -----------------------------------------------------------------------
    # Main evaluation
    # -----------------------------------------------------------------------

    def evaluate_week(
        self,
        week: int,
        all_node_states: dict,
        guardrail_log: list,
    ) -> list[Alert]:
        """
        Generate all alerts for the current simulation week.

        Parameters
        ----------
        week             : current simulation week number.
        all_node_states  : dict keyed by node name, each value is a dict of
                           KPIs/signals for that node this week.
        guardrail_log    : list of guardrail result dicts produced by
                           SymbolicGuardrailShield.project_onto_physics().

        Returns
        -------
        List of new Alert objects raised this week (deduplicated).
        """
        alerts: list[Alert] = []

        alerts += self._check_demand_signals(week, all_node_states)
        alerts += self._check_return_wave(week, all_node_states)
        alerts += self._check_inventory_health(week, all_node_states)
        alerts += self._check_seasonal_risk(week, all_node_states)
        alerts += self._check_financial_risk(week, all_node_states)
        alerts += self._check_macro_regime(week, all_node_states)
        alerts += self._from_guardrail_log(week, guardrail_log)

        alerts = self._deduplicate(alerts, week)

        # Expire alerts older than 4-week rolling window
        self.active_alerts = [a for a in self.active_alerts if a.week >= week - 4]
        self.active_alerts.extend(alerts)
        self.alert_history.extend(alerts)

        return alerts

    # -----------------------------------------------------------------------
    # Demand signals
    # -----------------------------------------------------------------------

    def _check_demand_signals(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        store = states.get("store", {})

        gross_demand = float(store.get("gross_demand", 0))
        organic_baseline = float(store.get("organic_demand_baseline", gross_demand or 1))
        avg_4w_demand = float(store.get("avg_4w_demand", organic_baseline or 1))

        # ---- Check 1: Promo spike (demand > 2.5× organic baseline) ------
        if organic_baseline > 0 and gross_demand > 2.5 * organic_baseline:
            ratio = gross_demand / organic_baseline
            alerts.append(self._make(
                node="store",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.DEMAND,
                title="Promotional Demand Spike Detected",
                detail=(
                    f"Gross demand ({gross_demand:.0f} units) is {ratio:.1f}× the "
                    f"organic baseline ({organic_baseline:.0f} units). "
                    "Promotional lift may mask underlying sell-through weakness "
                    "and will generate a return wave in 10–14 days."
                ),
                action=(
                    "Decompose demand into organic / promo / panic components via the "
                    "PHANTOM_DEMAND_SPLIT signal before issuing a replenishment order. "
                    "Pre-arm GR1 return-wave netting on the central warehouse node. "
                    "Do NOT place incremental factory orders until returns settle."
                ),
                metric_name="demand_to_organic_ratio",
                metric_value=round(ratio, 2),
                threshold=2.5,
                week=week,
            ))

        # ---- Check 2: Demand crash (< −30 % of 4-week average) ----------
        if avg_4w_demand > 0:
            change_pct = (gross_demand - avg_4w_demand) / avg_4w_demand
            if change_pct < -0.30:
                alerts.append(self._make(
                    node="store",
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.DEMAND,
                    title="Demand Crash — Possible Monsoon Dip or Broken Size Curve",
                    detail=(
                        f"This week's demand ({gross_demand:.0f} units) is "
                        f"{abs(change_pct):.0%} below the 4-week average "
                        f"({avg_4w_demand:.0f} units). Likely causes: monsoon "
                        "footfall drop, broken M/L size curve, or competitive promo."
                    ),
                    action=(
                        "Audit SIZE_CURVE_HEALTH signal for M/L stockouts. "
                        "Check POS_VELOCITY by size to isolate root cause. "
                        "If size curve is broken, activate GR3 lateral transfer. "
                        "Hold replenishment orders until demand stabilises for 2 weeks."
                    ),
                    metric_name="demand_change_vs_4w_avg",
                    metric_value=round(change_pct, 4),
                    threshold=-0.30,
                    week=week,
                ))

        return alerts

    # -----------------------------------------------------------------------
    # Return wave
    # -----------------------------------------------------------------------

    def _check_return_wave(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        wh = states.get("central_warehouse", {})

        predicted_returns = float(wh.get("predicted_returns_10day", 0))
        weekly_shipments = float(wh.get("weekly_shipments", 1) or 1)
        intake_capacity = float(wh.get("weekly_intake_capacity", 9999))
        flash_sale_return_rate = float(wh.get("flash_sale_return_rate", 0))

        return_rate = predicted_returns / weekly_shipments if weekly_shipments > 0 else 0.0

        # ---- Check 1: Return wave > 35 % of weekly shipments ------------
        if return_rate > 0.35:
            alerts.append(self._make(
                node="central_warehouse",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.RETURNS,
                title="High Return Wave — Phantom Inventory Risk",
                detail=(
                    f"Predicted 10-day return wave ({predicted_returns:.0f} units) "
                    f"is {return_rate:.0%} of weekly shipments ({weekly_shipments:.0f} "
                    "units). GR1 netting will deduct these from the factory order, "
                    "but the physical processing backlog will stress the WMS."
                ),
                action=(
                    "Activate GR1 Return-Wave Netting immediately — ensure predicted "
                    "returns are deducted before any factory order is issued. "
                    "Alert WMS team to prepare returns intake lane. "
                    "Do not treat incoming returns as new demand fulfillment stock "
                    "until QC-cleared."
                ),
                metric_name="return_rate_vs_shipments",
                metric_value=round(return_rate, 4),
                threshold=0.35,
                week=week,
            ))

        # ---- Check 2: Flash-sale return rate > 42 % ---------------------
        if flash_sale_return_rate > 0.42:
            alerts.append(self._make(
                node="central_warehouse",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.RETURNS,
                title="Flash-Sale Return Rate Spiking",
                detail=(
                    f"Flash-sale return rate is {flash_sale_return_rate:.0%}, "
                    "exceeding the 42 % alert threshold. This indicates either "
                    "product-description mismatch, size curve breakage, or "
                    "predatory buy-return behaviour."
                ),
                action=(
                    "Flag top-10 returning SKUs to Merchandising for product page audit. "
                    "Consider adding a 'Size Guide' prompt at checkout for flagged SKUs. "
                    "Discuss flash-sale return policy tightening with Brand team. "
                    "Re-evaluate flash-sale demand as non-organic in the stats model."
                ),
                metric_name="flash_sale_return_rate",
                metric_value=round(flash_sale_return_rate, 4),
                threshold=0.42,
                week=week,
            ))

        # ---- Check 3: Return wave exceeds intake capacity ----------------
        if predicted_returns > intake_capacity * 0.90:
            overflow_units = predicted_returns - intake_capacity
            alerts.append(self._make(
                node="central_warehouse",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.RETURNS,
                title="CRITICAL — Return Wave Will Exceed Warehouse Intake Capacity",
                detail=(
                    f"Predicted returns ({predicted_returns:.0f} units) will "
                    f"exceed 90 % of intake capacity ({intake_capacity:.0f} units/week). "
                    f"Estimated overflow: ~{max(0, overflow_units):.0f} units. "
                    "Physical throughput collapse risk."
                ),
                action=(
                    "Activate overflow returns dock at secondary 3PL facility immediately. "
                    "Delay outbound dispatch by 48 h to clear intake queue. "
                    "Alert returns QC team to run triage priority processing. "
                    "Notify Operations Head — may require weekend overtime."
                ),
                metric_name="return_wave_vs_intake_capacity",
                metric_value=round(predicted_returns, 0),
                threshold=intake_capacity * 0.90,
                week=week,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Inventory health
    # -----------------------------------------------------------------------

    def _check_inventory_health(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        store = states.get("store", {})
        wh = states.get("central_warehouse", {})

        # Store-level
        effective_inv = float(store.get("effective_inventory_weeks", 99))
        flagship_ml = float(store.get("flagship_ML_completeness", 1.0))

        # Warehouse-level
        fill_rate = float(wh.get("fill_rate", 1.0))
        bullwhip_ratio = float(wh.get("bullwhip_ratio_node2", 0.0))

        # ---- Check 1: Effective inventory < 1 week cover ----------------
        if effective_inv < 1.0:
            alerts.append(self._make(
                node="store",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.INVENTORY,
                title="CRITICAL — Store Cover Below 1 Week",
                detail=(
                    f"Effective inventory cover is {effective_inv:.2f} weeks "
                    "(below 1-week minimum). Stockout risk is imminent. "
                    "Lost-sales opportunity cost is accumulating each day."
                ),
                action=(
                    "Trigger emergency replenishment from central warehouse today. "
                    "If WH stock is insufficient, activate inter-store lateral transfer "
                    "for best-selling sizes. Escalate to Store Operations Head. "
                    "Consider emergency purchase order to mill on fast-track."
                ),
                metric_name="effective_inventory_weeks",
                metric_value=round(effective_inv, 2),
                threshold=1.0,
                week=week,
            ))

        # ---- Check 2: M/L missing from flagship -------------------------
        if flagship_ml < 0.30:
            alerts.append(self._make(
                node="store",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.INVENTORY,
                title="M/L Size Curve Broken at Flagship Stores",
                detail=(
                    f"Flagship store M/L rack completeness is {flagship_ml:.0%} "
                    "(threshold: 30 %). M and L sizes account for ~55 % of Indian "
                    "apparel volume. Broken size curve directly suppresses conversion."
                ),
                action=(
                    "Check slow_store_ML_excess — if > 50 units, activate GR3 "
                    "Lateral Size Consolidation to reroute without factory reorder. "
                    "If lateral transfer insufficient, raise priority replenishment "
                    "request to central warehouse for M/L sizes only."
                ),
                metric_name="flagship_ML_completeness",
                metric_value=round(flagship_ml, 4),
                threshold=0.30,
                week=week,
            ))

        # ---- Check 3: WH fill rate < 85 % -------------------------------
        if fill_rate < 0.85:
            alerts.append(self._make(
                node="central_warehouse",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.INVENTORY,
                title="Warehouse Fill Rate Below 85 % — Store Orders Unfulfilled",
                detail=(
                    f"Central warehouse fill rate this week: {fill_rate:.0%} "
                    "(threshold: 85 %). Stores are receiving partial shipments, "
                    "compounding size-curve breakage and conversion loss."
                ),
                action=(
                    "Identify top-5 unfulfilled SKUs and expedite from mill. "
                    "Review pipeline orders for delivery acceleration options. "
                    "Communicate partial-fulfillment ETA to store operations. "
                    "If pattern persists 2+ weeks, review safety stock parameters."
                ),
                metric_name="warehouse_fill_rate",
                metric_value=round(fill_rate, 4),
                threshold=0.85,
                week=week,
            ))

        # ---- Check 4: Bullwhip ratio > 2.0× at Node 2 ------------------
        if bullwhip_ratio > 2.0:
            alerts.append(self._make(
                node="central_warehouse",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.INVENTORY,
                title="Bullwhip Amplification Exceeds 2× at Central Warehouse",
                detail=(
                    f"Order variance ratio at Node 2 is {bullwhip_ratio:.2f}× "
                    "the organic demand variance (GR2 limit: 1.15×). "
                    "Upstream orders are significantly amplifying downstream demand "
                    "signals — classic bullwhip effect in progress."
                ),
                action=(
                    "Verify GR2 Bullwhip Slew-Rate Ceiling is armed and applying "
                    "correctly. Review last 8 weeks of order history for anomalies. "
                    "Freeze any discretionary safety-stock build this week. "
                    "Present bullwhip chart to Planning team for root-cause review."
                ),
                metric_name="bullwhip_ratio_node2",
                metric_value=round(bullwhip_ratio, 2),
                threshold=2.0,
                week=week,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Seasonal risk
    # -----------------------------------------------------------------------

    def _check_seasonal_risk(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        mill = states.get("mill", {})

        season_weeks_remaining = int(mill.get("season_weeks_remaining", 99))
        greige_batches = float(mill.get("greige_batches_available", 99))
        cliff_active = bool(mill.get("eoss_cliff_active", False))

        # ---- Check 1: EOSS cliff activated (≤ 8 weeks) ------------------
        if season_weeks_remaining <= 8 or cliff_active:
            alerts.append(self._make(
                node="mill",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SEASONAL,
                title="CRITICAL — EOSS Suicide Window Active",
                detail=(
                    f"Only {season_weeks_remaining} weeks remain until season close "
                    f"(threshold: 8 weeks ≈ 52-day total lead time + 2-week buffer). "
                    "GR4 has zeroed all new fabric orders. Any new production cannot "
                    "reach stores before the season ends — deadstock risk is severe."
                ),
                action=(
                    "IMMEDIATELY switch all mill activity to greige conversion only. "
                    "Prioritise bestseller SKUs for the remaining conversion capacity. "
                    "Co-ordinate with Merchandising on EOSS markdown calendar. "
                    "Alert CFO: no new capex on fabric this season."
                ),
                metric_name="season_weeks_remaining",
                metric_value=float(season_weeks_remaining),
                threshold=8.0,
                week=week,
                guardrail_triggered="GR4_EOSS_SUICIDE_WINDOW",
                auto_resolved=True,
            ))

        # ---- Check 2: Season ≤ 14 weeks (warm warning) ------------------
        elif season_weeks_remaining <= 14:
            alerts.append(self._make(
                node="mill",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.SEASONAL,
                title="Season End Approaching — 14-Week Warm Warning",
                detail=(
                    f"{season_weeks_remaining} weeks to season close. "
                    "Lead time buffer is narrowing — new fabric orders placed after "
                    "week 8 cannot be sold this season. Now is the last window to "
                    "plan remaining production."
                ),
                action=(
                    "Finalise fabric orders for the season within next 3 weeks. "
                    "Begin greige stock audit — identify conversion-ready lots. "
                    "Lock in bestseller replenishment quantities with mill today. "
                    "Prepare EOSS markdown plan with Merchandising."
                ),
                metric_name="season_weeks_remaining",
                metric_value=float(season_weeks_remaining),
                threshold=14.0,
                week=week,
            ))

        # ---- Check 3: Greige < 2 batches when conversion-only -----------
        if cliff_active and greige_batches < 2.0:
            alerts.append(self._make(
                node="mill",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SEASONAL,
                title="CRITICAL — Greige Stock Critically Low in Conversion-Only Mode",
                detail=(
                    f"EOSS cliff is active (conversion-only) but greige stock is only "
                    f"{greige_batches:.1f} batches (threshold: 2 batches). "
                    "Conversion capacity will be idle; no new fabric can be ordered. "
                    "Season fulfilment at severe risk."
                ),
                action=(
                    "Immediately source greige fabric from secondary supplier on spot basis. "
                    "Evaluate emergency inter-mill transfer of greige from partner mill. "
                    "Prioritise remaining greige for highest-margin, fastest-selling SKUs. "
                    "Escalate to Supply Chain Director — potential lost-sales impact this week."
                ),
                metric_name="greige_batches_available",
                metric_value=round(greige_batches, 2),
                threshold=2.0,
                week=week,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Financial risk
    # -----------------------------------------------------------------------

    def _check_financial_risk(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        proc = states.get("procurement", {})

        proposed_mrp = int(proc.get("proposed_mrp", 0))
        cost_pressure_index = float(proc.get("cost_pressure_index", 1.0))
        hedge_pct = float(proc.get("hedged_yarn_pct", 1.0))

        # ---- Check 1: MRP in GST cliff band ₹1001–₹1085 ----------------
        if 1001 <= proposed_mrp <= 1085:
            alerts.append(self._make(
                node="procurement",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.FINANCIAL,
                title="CRITICAL — Proposed MRP Enters 12 % GST Cliff Band",
                detail=(
                    f"Proposed MRP ₹{proposed_mrp:,} falls in the ₹1,001–₹1,085 "
                    "GST 12 % band. This is above the ₹999 threshold where consumers "
                    "pay 5 % GST. GR5 will auto-lock MRP at ₹999, but the resulting "
                    "margin gap must be closed via procurement action."
                ),
                action=(
                    "Switch immediately to hedged yarn lot procurement to recover margin. "
                    "Engage CFO on whether to absorb margin gap or reduce pack-size count. "
                    "Explore ₹799 value-pack option as alternative revenue path. "
                    "GR5 is auto-applying the ₹999 lock — confirm in the guardrail trace."
                ),
                metric_name="proposed_mrp",
                metric_value=float(proposed_mrp),
                threshold=1001.0,
                week=week,
                guardrail_triggered="GR5_GST_CLIFF_SHIELD",
                auto_resolved=True,
            ))

        # ---- Check 2: Cost pressure index > 1.15 -------------------------
        if cost_pressure_index > 1.15:
            alerts.append(self._make(
                node="procurement",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.FINANCIAL,
                title="Input Cost Pressure Index Exceeding 1.15× — Margin Squeeze",
                detail=(
                    f"Cost pressure index is {cost_pressure_index:.2f}× the baseline "
                    "(threshold: 1.15×). Rising yarn / cotton / freight costs are "
                    "compressing gross margin. If sustained, will force MRP into the "
                    "GST cliff band."
                ),
                action=(
                    "Increase hedged yarn ratio to at least 70 % of procurement volume. "
                    "Review freight routing for cost optimisation opportunities. "
                    "Escalate to CFO for pre-approval of hedge extension. "
                    "Model MRP sensitivity: at what index does MRP breach ₹1,000?"
                ),
                metric_name="cost_pressure_index",
                metric_value=round(cost_pressure_index, 4),
                threshold=1.15,
                week=week,
            ))

        # ---- Check 3: Hedged yarn < 60 % --------------------------------
        if hedge_pct < 0.60:
            alerts.append(self._make(
                node="procurement",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.FINANCIAL,
                title="Hedged Yarn Below 60 % — Spot Price Exposure",
                detail=(
                    f"Only {hedge_pct:.0%} of yarn procurement is hedged "
                    "(minimum target: 60 %). Higher spot-market exposure increases "
                    "cost volatility and could rapidly push MRP toward the GST cliff "
                    "if cotton index spikes."
                ),
                action=(
                    "Contact approved yarn suppliers to lock additional hedged volume. "
                    "Target: bring hedged ratio to ≥ 65 % by end of this week. "
                    "Review forward cotton price curve — opportunistic hedge if prices dip. "
                    "Flag current exposure in weekly P&L review."
                ),
                metric_name="hedged_yarn_pct",
                metric_value=round(hedge_pct, 4),
                threshold=0.60,
                week=week,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Macro regime
    # -----------------------------------------------------------------------

    def _check_macro_regime(self, week: int, states: dict) -> list[Alert]:
        alerts: list[Alert] = []
        proc = states.get("procurement", {})
        mill = states.get("mill", {})

        freight_index = float(proc.get("freight_congestion_index", 100))
        cotton_index = float(proc.get("cotton_index", 100))
        lead_time_estimate = float(mill.get("lead_time_estimate_days", 52))

        # ---- Check 1: Freight congestion > 130 --------------------------
        if freight_index > 130:
            expanded_lead = lead_time_estimate * (freight_index / 100)
            alerts.append(self._make(
                node="mill",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.MACRO,
                title="Freight Congestion Index > 130 — Lead Time Expansion Risk",
                detail=(
                    f"Freight congestion index is {freight_index:.0f} "
                    "(threshold: 130). Current lead time estimate of "
                    f"{lead_time_estimate:.0f} days may expand to "
                    f"~{expanded_lead:.0f} days under current congestion. "
                    "EOSS cliff calculations should be adjusted accordingly."
                ),
                action=(
                    "Update lead_time_estimate in the physics state to reflect "
                    "freight congestion adjustment. Re-run EOSS cliff check with "
                    "revised estimate — GR4 may trigger earlier. "
                    "Evaluate air freight for critical top-selling SKUs. "
                    "Alert Planning to freeze any orders dependent on 52-day lead time."
                ),
                metric_name="freight_congestion_index",
                metric_value=round(freight_index, 1),
                threshold=130.0,
                week=week,
            ))

        # ---- Check 2: Cotton index > 105 (routes to procurement only) ---
        if cotton_index > 105:
            alerts.append(self._make(
                node="procurement",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.MACRO,
                title="Cotton Price Index Above 105 — Long-Run Cost Pressure Building",
                detail=(
                    f"Cotton price index is {cotton_index:.1f} "
                    "(alert threshold: 105). Sustained elevation will feed through "
                    "to yarn costs within 4–8 weeks, increasing the cost_pressure_index "
                    "and potentially triggering the GST cliff guardrail."
                ),
                action=(
                    "Monitor cotton index weekly — if sustained above 108 for 3 weeks, "
                    "escalate to CFO for emergency hedge review. "
                    "Pre-model MRP impact at cotton index = 110 and 115. "
                    "Consider accelerating hedged yarn procurement at current price levels."
                ),
                metric_name="cotton_index",
                metric_value=round(cotton_index, 1),
                threshold=105.0,
                week=week,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Guardrail log → alerts
    # -----------------------------------------------------------------------

    def _from_guardrail_log(self, week: int, guardrail_log: list) -> list[Alert]:
        """
        Convert guardrail trigger events from SymbolicGuardrailShield into
        GUARDRAIL-category Alert objects.

        Each entry in *guardrail_log* is expected to be a dict with at minimum:
            rule      : str   — e.g. "GR1_RETURN_WAVE_NETTING"
            triggered : bool
            node      : str
            trace     : str
            savings   : str
            final_order : int
        """
        alerts: list[Alert] = []

        _rule_meta: dict[str, tuple[AlertSeverity, str]] = {
            "GR1_RETURN_WAVE_NETTING": (
                AlertSeverity.HIGH,
                "Deduct predicted returns before placing factory order.",
            ),
            "GR2_BULLWHIP_SLEW_RATE": (
                AlertSeverity.MEDIUM,
                "Review order history for bullwhip amplification patterns.",
            ),
            "GR3_LATERAL_SIZE_CONSOLIDATION": (
                AlertSeverity.HIGH,
                "Confirm lateral transfer logistics with warehouse team.",
            ),
            "GR4_EOSS_SUICIDE_WINDOW": (
                AlertSeverity.CRITICAL,
                "Switch mill to greige-conversion-only mode immediately.",
            ),
            "GR5_GST_CLIFF_SHIELD": (
                AlertSeverity.CRITICAL,
                "Confirm MRP locked at ₹999 and switch to hedged yarn procurement.",
            ),
        }

        for entry in guardrail_log:
            if not entry.get("triggered", False):
                continue

            rule = entry.get("rule", "UNKNOWN")
            node = entry.get("node", "unknown")
            trace = entry.get("trace", "")
            savings = entry.get("savings", "")
            final_order = entry.get("final_order", 0)

            severity, default_action = _rule_meta.get(
                rule, (AlertSeverity.MEDIUM, "Review guardrail trace log.")
            )

            alerts.append(self._make(
                node=node,
                severity=severity,
                category=AlertCategory.GUARDRAIL,
                title=f"Guardrail Triggered — {rule}",
                detail=(
                    f"{trace} "
                    f"Final order resolved to {final_order} units. "
                    f"Estimated savings: {savings}."
                ),
                action=default_action,
                metric_name="guardrail_activation",
                metric_value=1.0,
                threshold=1.0,
                week=week,
                guardrail_triggered=rule,
                auto_resolved=True,
            ))

        return alerts

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def _deduplicate(self, new_alerts: list[Alert], current_week: int) -> list[Alert]:
        """
        Suppress alerts with the same (category, node, metric_name) raised
        within the last _DEDUP_WINDOW_WEEKS weeks.
        """
        # Build set of recent dedup keys from active alerts within the window
        recent_keys: set[str] = {
            a.dedup_key
            for a in self.active_alerts
            if current_week - a.week < self._DEDUP_WINDOW_WEEKS
        }

        deduped: list[Alert] = []
        seen_this_batch: set[str] = set()

        for alert in new_alerts:
            key = alert.dedup_key
            if key in recent_keys or key in seen_this_batch:
                continue
            deduped.append(alert)
            seen_this_batch.add(key)

        return deduped

    # -----------------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------------

    def get_node_alerts(
        self,
        node: str,
        severity_min: AlertSeverity = AlertSeverity.LOW,
    ) -> list[Alert]:
        """
        Return all active alerts for *node*, filtered to *severity_min* or worse,
        sorted by severity (CRITICAL first).

        Parameters
        ----------
        node         : supply chain node name.
        severity_min : minimum severity to include.
        """
        order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
        }
        min_order = order[severity_min]
        filtered = [
            a for a in self.active_alerts
            if a.node == node and order[a.severity] <= min_order
        ]
        return sorted(filtered, key=lambda a: order[a.severity])

    def get_dashboard_summary(self) -> dict:
        """
        Aggregated alert counts and top priorities for the dashboard header row.

        Returns
        -------
        dict with:
            total_active         : int
            by_severity          : dict[severity_label → count]
            by_category          : dict[category_label → count]
            by_node              : dict[node → count]
            top_critical         : list of (node, title) for CRITICAL alerts
            top_high             : list of (node, title) for HIGH alerts
        """
        by_sev: dict[str, int] = {s.value: 0 for s in AlertSeverity}
        by_cat: dict[str, int] = {c.value: 0 for c in AlertCategory}
        by_node: dict[str, int] = {}

        for a in self.active_alerts:
            by_sev[a.severity.value] += 1
            by_cat[a.category.value] += 1
            by_node[a.node] = by_node.get(a.node, 0) + 1

        critical = [
            {"node": a.node, "title": a.title, "id": a.id}
            for a in self.active_alerts
            if a.severity == AlertSeverity.CRITICAL
        ]
        high = [
            {"node": a.node, "title": a.title, "id": a.id}
            for a in self.active_alerts
            if a.severity == AlertSeverity.HIGH
        ]

        return {
            "total_active": len(self.active_alerts),
            "by_severity": by_sev,
            "by_category": by_cat,
            "by_node": by_node,
            "top_critical": critical,
            "top_high": high,
        }

    def dismiss(self, alert_id: str) -> bool:
        """
        Remove an alert by ID from the active alerts list.

        Returns True if dismissed, False if not found.
        """
        before = len(self.active_alerts)
        self.active_alerts = [a for a in self.active_alerts if a.id != alert_id]
        return len(self.active_alerts) < before

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AlertEngine("
            f"active={len(self.active_alerts)}, "
            f"history={len(self.alert_history)})"
        )
