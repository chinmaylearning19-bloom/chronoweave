"""
simulation_engine.py — ChronoWeave Master Orchestrator
Wires all components: Data → Kalman Stats → Physics → Guardrails → Info Bus → Alerts → LLM Chat

Node topology (Indian Omnichannel Apparel):
  Node 1 (store)             : 20 retail stores + D2C online channel
  Node 2 (central_warehouse) : Central distribution hub, the critical amplification point
  Node 3 (mill)              : Fabric mill (cut-make-trim + dye-vat)
  Node 4 (procurement)       : Yarn/greige procurement and costing

Per-week simulation loop:
  1. Generate / load 3-layer data for week t
  2. Run Kalman Filter → clean organic demand estimate
  3. Run Beta-Binomial Return Estimator → predict return wave
  4. Run Gamma Lead-Time Estimator → update pipeline delays
  5. Compute LegacyERP policy (baseline comparison)
  6. Compute proposed neurosymbolic orders
  7. Run Wardenclyffe physics per node
  8. Apply Symbolic Guardrails → final orders
  9. Publish all state to Information Bus
  10. Run Alert Engine → generate alerts
  11. Return full week payload to API
"""

import sys, os, math, random
from collections import defaultdict, deque
from typing import Optional

# Path setup
_here = os.path.dirname(__file__)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, '..', 'data'))

from wardenclyffe_core import NodePhysicsEngine


# ── Lazy imports (allow partial import if submodules still generating) ──────────
def _lazy_import():
    from adaptive_stats import KalmanDemandFilter, BetaBinomialReturnEstimator, GammaLeadTimeEstimator
    from symbolic_guardrails import SymbolicGuardrailShield
    from information_bus import InformationBus, NodeMessage
    from alert_system import AlertEngine, AlertSeverity
    from node_chat import NodeChatSession
    from data_generator import generate_dataset, generate_legacy_erp_orders
    return (KalmanDemandFilter, BetaBinomialReturnEstimator, GammaLeadTimeEstimator,
            SymbolicGuardrailShield, InformationBus, NodeMessage,
            AlertEngine, AlertSeverity, NodeChatSession,
            generate_dataset, generate_legacy_erp_orders)


class NodeInventoryState:
    """Simple inventory ledger per node."""

    def __init__(self, name: str, initial_inventory: int, lead_time_days: float, capacity: int):
        self.name             = name
        self.inventory        = initial_inventory
        self.backlog          = 0
        self.pipeline: deque  = deque([0] * max(1, round(lead_time_days / 7)), maxlen=20)
        self.orders_placed:list = []
        self.shipments_sent:list = []
        self.inventory_history:list = []
        self.backlog_history:list  = []
        self.cost_history:list     = []
        self.capacity             = capacity
        self.total_holding_cost   = 0.0
        self.total_backorder_cost = 0.0

    def receive_shipment(self) -> int:
        return int(self.pipeline[0]) if self.pipeline else 0

    def step(self, downstream_order: int, upstream_order: int,
             holding_cost: float = 5.0, backorder_cost: float = 25.0) -> dict:
        arrived      = self.receive_shipment()
        self.inventory += arrived
        fulfil        = min(downstream_order + self.backlog, self.inventory)
        self.inventory -= fulfil
        self.backlog   = max(0, downstream_order + self.backlog - fulfil)
        self.pipeline.append(upstream_order)

        hc = self.inventory * holding_cost
        bc = self.backlog   * backorder_cost
        self.total_holding_cost   += hc
        self.total_backorder_cost += bc
        self.inventory_history.append(self.inventory)
        self.backlog_history.append(self.backlog)
        self.cost_history.append({"holding": hc, "backorder": bc})
        self.orders_placed.append(upstream_order)
        self.shipments_sent.append(fulfil)

        return {"inventory": self.inventory, "backlog": self.backlog, "fulfilled": fulfil, "arrived": arrived}

    def get_state(self, week: int = 0) -> dict:
        return {
            "node":        self.name,
            "week":        week,
            "inventory":   self.inventory,
            "backlog":     self.backlog,
            "pipeline":    list(self.pipeline),
            "fill_rate":   round(self.shipments_sent[-1] / max(self.orders_placed[-2] if len(self.orders_placed) > 1 else 1, 1), 3) if self.orders_placed else 1.0,
            "total_holding_cost":   round(self.total_holding_cost, 2),
            "total_backorder_cost": round(self.total_backorder_cost, 2),
        }


class LegacyERPPolicy:
    """
    Baseline naive ERP: 4-week moving average of immediate downstream orders + 15% buffer + MOQ.
    Should reproduce 3.4x-5.9x bullwhip amplification.
    """
    def __init__(self, moq: int = 500):
        self.order_history: deque = deque(maxlen=4)
        self.moq = moq

    def decide(self, downstream_order: int) -> int:
        self.order_history.append(downstream_order)
        avg = sum(self.order_history) / len(self.order_history)
        raw = avg * 1.15  # 15% safety buffer
        # MOQ rounding up
        return math.ceil(raw / self.moq) * self.moq


class ChronoWeaveEngine:
    """
    Full ChronoWeave simulation engine — 104-week Indian omnichannel apparel supply chain.
    """

    NODES = ["store", "central_warehouse", "mill", "procurement"]

    def __init__(self, sector: str = "apparel_india"):
        self.sector  = sector
        self.week    = 0
        self.dataset: Optional[object] = None  # pd.DataFrame once loaded
        self._modules_loaded = False

        # Inventory nodes
        self.nodes = {
            "store":             NodeInventoryState("store",             initial_inventory=8000,  lead_time_days=3,  capacity=15000),
            "central_warehouse": NodeInventoryState("central_warehouse", initial_inventory=18000, lead_time_days=14, capacity=50000),
            "mill":              NodeInventoryState("mill",              initial_inventory=35000, lead_time_days=35, capacity=120000),
            "procurement":       NodeInventoryState("procurement",       initial_inventory=80000, lead_time_days=21, capacity=300000),
        }

        # Physics engines
        self.physics_engines = {n: NodePhysicsEngine(n) for n in self.NODES}

        # Legacy ERP policies (per node, for A/B comparison)
        self.legacy_erp = {n: LegacyERPPolicy(moq=500 if n == "mill" else 1) for n in self.NODES}

        # Simulation log
        self.sim_log: list = []
        self.erp_log: list = []

        # Chat sessions
        self._chat: Optional[object] = None

        # Greige stock (batches of 200 units)
        self.greige_stock_batches = 40

        # Season tracking
        self.current_season_end = 26

    def _load_modules(self):
        if self._modules_loaded:
            return
        (self.KalmanDemandFilter, self.BetaBinomialReturnEstimator,
         self.GammaLeadTimeEstimator, self.SymbolicGuardrailShield,
         self.InformationBus, self.NodeMessage,
         self.AlertEngine, self.AlertSeverity,
         self.NodeChatSession, self.generate_dataset,
         self.generate_legacy_erp_orders) = _lazy_import()

        self.kalman      = self.KalmanDemandFilter(base_demand=2200.0)
        self.return_est  = self.BetaBinomialReturnEstimator()
        self.lead_est    = {
            "store":             self.GammaLeadTimeEstimator(base_days=3.0,  eta=0.02, node_id="store"),
            "central_warehouse": self.GammaLeadTimeEstimator(base_days=14.0, eta=0.05, node_id="central_warehouse"),
            "mill":              self.GammaLeadTimeEstimator(base_days=35.0, eta=0.08, node_id="mill"),
        }
        self.guardrails  = self.SymbolicGuardrailShield()
        self.info_bus    = self.InformationBus()
        self.alert_engine = self.AlertEngine(self.info_bus)
        self._chat       = self.NodeChatSession()
        self.dataset     = self.generate_dataset(seed=42)
        self._modules_loaded = True

    def _get_week_data(self, week: int) -> dict:
        """Extract week's data from the 104-week dataset (1-indexed)."""
        if self.dataset is None:
            return {}
        idx = min(week - 1, len(self.dataset) - 1)
        row = self.dataset.iloc[idx]
        return row.to_dict()

    def step(self) -> dict:
        """Advance simulation by one week. Returns full payload."""
        self._load_modules()
        self.week += 1
        w_data = self._get_week_data(self.week)

        # ── Layer 1: Macro ────────────────────────────────────────────────────
        macro = {
            "cotton_index":           w_data.get("cotton_index", 85.0),
            "freight_congestion_index": w_data.get("freight_congestion_index", 100.0),
            "brent_crude_inr":        w_data.get("brent_crude_inr", 6500.0),
        }

        # ── Layer 2: Micro Demand ─────────────────────────────────────────────
        gross_demand      = float(w_data.get("gross_demand_units", 2200.0))
        promo_flag        = int(w_data.get("promo_flag", 0))
        online_share      = float(w_data.get("online_share", 0.35))
        rack_completeness = float(w_data.get("rack_completeness", 0.95))
        ml_completeness   = float(w_data.get("size_curve_M", 0.22) + w_data.get("size_curve_L", 0.28)) / 0.50

        # ── Kalman Filter: Clean Demand Estimate ──────────────────────────────
        kalman_out = self.kalman.update(gross_demand, rack_completeness, promo_flag)
        organic_demand  = kalman_out.get("organic_demand", gross_demand * 0.7)
        promo_spike     = kalman_out.get("promo_lift", 0.0)
        clean_forecast  = self.kalman.get_clean_forecast()

        # ── Beta-Binomial: Return Wave Prediction ─────────────────────────────
        online_shipped = gross_demand * online_share
        self.return_est.update(int(online_shipped), max(0, int(online_shipped * (0.42 if promo_flag else 0.28))), promo_flag)
        online_shipped_history = list(self.nodes["central_warehouse"].pipeline)
        return_wave = self.return_est.predict_return_wave(online_shipped_history)
        predicted_returns_week = return_wave.get("expected_returns_week", 0)
        net_demand   = self.return_est.get_net_demand(organic_demand, predicted_returns_week)

        # ── Gamma Lead-Time: Pipeline Delay ──────────────────────────────────
        lead_estimates = {}
        total_lead_days = 0
        for nid, est in self.lead_est.items():
            lt = est.estimate(macro["freight_congestion_index"])
            lead_estimates[nid] = lt
            if nid != "store":
                total_lead_days += lt.get("mean_days", 0)

        season_weeks_remaining = self.current_season_end - self.week
        if season_weeks_remaining <= 0:
            self.current_season_end += 26  # next season
            season_weeks_remaining = self.current_season_end - self.week

        # ── Node States ───────────────────────────────────────────────────────
        node_states = {}

        node_states["store"] = {
            **self.nodes["store"].get_state(self.week),
            "gross_demand":        gross_demand,
            "organic_demand":      round(organic_demand, 1),
            "promo_flag":          promo_flag,
            "promo_spike_units":   round(promo_spike, 1),
            "net_demand":          round(net_demand, 1),
            "online_share":        online_share,
            "rack_completeness":   round(rack_completeness, 3),
            "ml_size_completeness":round(ml_completeness, 3),
            "pos_velocity":        round(gross_demand / 7, 1),
            "size_curve_M":        w_data.get("size_curve_M", 0.22),
            "size_curve_L":        w_data.get("size_curve_L", 0.28),
            "season_weeks_remaining": season_weeks_remaining,
        }

        node_states["central_warehouse"] = {
            **self.nodes["central_warehouse"].get_state(self.week),
            "predicted_returns_week": round(predicted_returns_week, 0),
            "return_wave_forecast":   return_wave,
            "effective_inventory":    max(0, self.nodes["central_warehouse"].inventory - predicted_returns_week),
            "phantom_demand_split": {
                "organic":     round(organic_demand, 1),
                "promo_spike": round(promo_spike, 1),
                "return_offset": round(predicted_returns_week, 1),
                "net_replenishment_needed": round(net_demand, 1),
            },
            "promo_flag":          promo_flag,
            "season_weeks_remaining": season_weeks_remaining,
        }

        node_states["mill"] = {
            **self.nodes["mill"].get_state(self.week),
            "greige_stock_batches":   self.greige_stock_batches,
            "lead_time_estimate_days": lead_estimates.get("mill", {}).get("mean_days", 35),
            "season_weeks_remaining":  season_weeks_remaining,
            "total_lead_time_weeks":   round(total_lead_days / 7, 1),
            "dye_batch_status":        "OK" if self.greige_stock_batches > 5 else "LOW",
            "promo_flag":              promo_flag,
        }

        node_states["procurement"] = {
            **self.nodes["procurement"].get_state(self.week),
            "cotton_index":          macro["cotton_index"],
            "freight_congestion_index": macro["freight_congestion_index"],
            "cost_pressure_index":   round(macro["cotton_index"] / 85.0, 3),
            "mrp_base":              w_data.get("mrp_base", 999.0),
            "mrp_status":            "GST_CLIFF_RISK" if 1001 <= w_data.get("mrp_base", 999) <= 1085 else "SAFE",
            "gst_cliff_status":      1001 <= w_data.get("mrp_base", 999) <= 1085,
            "yarn_hedge_pct":        min(1.0, 0.6 + random.gauss(0, 0.08)),
        }

        # ── Wardenclyffe Physics (per node) ───────────────────────────────────
        physics = {}
        for nid in self.NODES:
            physics[nid] = self.physics_engines[nid].evaluate(node_states[nid], macro)

        # ── Proposed Neurosymbolic Orders ─────────────────────────────────────
        # Neural order = clean_forecast + backlog recovery + safety margin
        cw_backlog     = node_states["central_warehouse"]["backlog"]
        proposed_orders = {
            "store":             int(max(0, clean_forecast * 1.05 + cw_backlog * 0.5)),
            "central_warehouse": int(max(0, net_demand * 1.10 + self.nodes["central_warehouse"].backlog * 0.7)),
            "mill":              int(max(0, net_demand * 1.15)),
            "procurement":       int(max(0, net_demand * 1.20)),
        }

        # ── Legacy ERP Orders (A/B Comparison) ───────────────────────────────
        erp_orders = {
            "store":             self.legacy_erp["store"].decide(int(gross_demand)),
            "central_warehouse": self.legacy_erp["central_warehouse"].decide(int(gross_demand)),
            "mill":              self.legacy_erp["mill"].decide(int(gross_demand)),
            "procurement":       self.legacy_erp["procurement"].decide(int(gross_demand)),
        }

        # ── Symbolic Guardrails → Final Orders ───────────────────────────────
        guardrail_results = {}
        final_orders      = {}
        guardrail_log     = []
        slow_store_ml_excess = max(0, int(self.nodes["store"].inventory * 0.15 * (1 - ml_completeness)))

        for nid in self.NODES:
            gs = self.guardrails.project_onto_physics(
                node            = nid,
                proposed_order  = proposed_orders[nid],
                physics_state   = {
                    **node_states[nid],
                    "predicted_returns_10day":    predicted_returns_week * 1.4,
                    "organic_demand_variance":    max(1.0, organic_demand * 0.05),
                    "slow_store_ml_excess":       slow_store_ml_excess,
                    "flagship_ml_completeness":   ml_completeness,
                    "season_weeks_remaining":     season_weeks_remaining,
                    "total_lead_time_weeks":      round(total_lead_days / 7, 1),
                    "proposed_mrp":               w_data.get("mrp_base", 999.0),
                },
                stats_forecast  = {"organic_demand": organic_demand, "clean_forecast": clean_forecast},
                macro           = macro,
            )
            guardrail_results[nid] = gs
            final_orders[nid]      = gs.get("final_order", proposed_orders[nid])
            node_states[nid]["factory_order"] = final_orders[nid]

            for item in gs.get("trace_log", []):
                if item.get("triggered"):
                    guardrail_log.append({**item, "node": nid})

        # ── Greige stock consumption ──────────────────────────────────────────
        mill_rules = guardrail_results.get("mill", {}).get("guardrails_triggered", [])
        if any("EOSS" in str(r) or "GREIGE" in str(r) for r in mill_rules):
            self.greige_stock_batches = max(0, self.greige_stock_batches - 2)
        else:
            # Normal greige replenishment
            self.greige_stock_batches = min(60, self.greige_stock_batches + random.randint(0, 3))

        # ── Step Inventory Nodes ──────────────────────────────────────────────
        node_step_results = {}
        node_step_results["store"]             = self.nodes["store"].step(int(gross_demand), final_orders["store"])
        node_step_results["central_warehouse"] = self.nodes["central_warehouse"].step(final_orders["store"], final_orders["central_warehouse"])
        node_step_results["mill"]              = self.nodes["mill"].step(final_orders["central_warehouse"], final_orders["mill"])
        node_step_results["procurement"]       = self.nodes["procurement"].step(final_orders["mill"], final_orders["procurement"])

        # ── Publish Granular Signals to Information Bus ───────────────────────
        self.latest_node_states = node_states
        self.latest_macro = macro
        self.latest_guardrail_log = guardrail_log

        # Store signals
        self.info_bus.publish(self.NodeMessage("store", "NET_DEMAND", {"net_demand": node_states["store"]["net_demand"]}, self.week))
        self.info_bus.publish(self.NodeMessage("store", "SIZE_CURVE_HEALTH", {"rack_completeness": node_states["store"]["rack_completeness"], "ml_size_completeness": node_states["store"]["ml_size_completeness"]}, self.week))
        self.info_bus.publish(self.NodeMessage("store", "PROMO_FLAG", {"promo_flag": bool(promo_flag)}, self.week))
        self.info_bus.publish(self.NodeMessage("store", "STOCKOUT_SIZES", {"stocked_out": ["M", "L"] if ml_completeness < 0.5 else []}, self.week))
        self.info_bus.publish(self.NodeMessage("store", "POS_VELOCITY", {"pos_velocity": node_states["store"]["pos_velocity"]}, self.week))

        # Central Warehouse signals
        self.info_bus.publish(self.NodeMessage("central_warehouse", "RETURN_WAVE_FORECAST", return_wave, self.week))
        self.info_bus.publish(self.NodeMessage("central_warehouse", "EFFECTIVE_INVENTORY", {"effective_inventory": node_states["central_warehouse"]["effective_inventory"]}, self.week))
        self.info_bus.publish(self.NodeMessage("central_warehouse", "PHANTOM_DEMAND_SPLIT", node_states["central_warehouse"]["phantom_demand_split"], self.week))
        self.info_bus.publish(self.NodeMessage("central_warehouse", "FILL_RATE", {"fill_rate": node_states["central_warehouse"]["fill_rate"]}, self.week))
        self.info_bus.publish(self.NodeMessage("central_warehouse", "FACTORY_ORDER_SIGNAL", {"factory_order": final_orders["central_warehouse"]}, self.week))

        # Mill signals
        self.info_bus.publish(self.NodeMessage("mill", "GREIGE_STOCK", {"greige_stock_batches": self.greige_stock_batches}, self.week))
        self.info_bus.publish(self.NodeMessage("mill", "PRODUCTION_QUEUE", {"queue_units": final_orders["mill"]}, self.week))
        self.info_bus.publish(self.NodeMessage("mill", "SEASON_PROXIMITY", {"season_weeks_remaining": season_weeks_remaining}, self.week))
        self.info_bus.publish(self.NodeMessage("mill", "DYE_BATCH_STATUS", {"status": node_states["mill"]["dye_batch_status"]}, self.week))
        self.info_bus.publish(self.NodeMessage("mill", "LEAD_TIME_ESTIMATE", {"lead_time_days": node_states["mill"]["lead_time_estimate_days"]}, self.week))

        # Procurement signals
        self.info_bus.publish(self.NodeMessage("procurement", "COTTON_PRICE_SIGNAL", {"cotton_index": macro["cotton_index"]}, self.week))
        self.info_bus.publish(self.NodeMessage("procurement", "YARN_HEDGE_STATUS", {"yarn_hedge_pct": node_states["procurement"]["yarn_hedge_pct"]}, self.week))
        self.info_bus.publish(self.NodeMessage("procurement", "GST_CLIFF_STATUS", {"gst_cliff_status": node_states["procurement"]["gst_cliff_status"], "mrp_base": node_states["procurement"]["mrp_base"]}, self.week))
        self.info_bus.publish(self.NodeMessage("procurement", "SUPPLIER_LEAD_TIME", {"lead_time_days": 21}, self.week))
        self.info_bus.publish(self.NodeMessage("procurement", "COST_PRESSURE_INDEX", {"cost_pressure_index": node_states["procurement"]["cost_pressure_index"]}, self.week))

        # Weekly general state snapshot
        for nid in self.NODES:
            self.info_bus.publish(self.NodeMessage(
                sender_node = nid,
                signal_type = f"{nid.upper()}_STATE",
                payload     = node_states[nid],
                week        = self.week,
                priority    = "HIGH" if physics[nid]["inference_mode"] in ["CRISIS_RESPONSE", "SYSTEMIC_LOCK"] else "NORMAL",
            ))

        # ── Alerts ────────────────────────────────────────────────────────────
        all_states_for_alerts = {nid: {**node_states[nid], **node_step_results.get(nid, {})} for nid in self.NODES}
        alerts_this_week = self.alert_engine.evaluate_week(self.week, all_states_for_alerts, guardrail_log)
        alerts_dicts     = [
            {"id": a.id, "node": a.node, "severity": a.severity.value,
             "category": a.category.value, "title": a.title, "detail": a.detail,
             "action": a.action, "week": a.week,
             "auto_resolved": a.auto_resolved}
            for a in alerts_this_week
        ]

        # ── Bullwhip Ratio Computation (Variance Ratio: Var(Orders) / Var(Net Demand)) ──
        past_demands = [entry["demand"] for entry in self.sim_log[-12:]] + [gross_demand]
        past_neuro_cw = [entry["final_orders"]["central_warehouse"] for entry in self.sim_log[-12:]] + [final_orders["central_warehouse"]]
        past_erp_cw   = [entry["erp_orders"]["central_warehouse"] for entry in self.sim_log[-12:]] + [erp_orders["central_warehouse"]]

        def _variance(lst):
            if len(lst) < 2: return 1.0
            mean = sum(lst) / len(lst)
            return sum((x - mean) ** 2 for x in lst) / max(len(lst) - 1, 1)

        var_demand = max(_variance(past_demands), 1.0)
        bw_neuro = _variance(past_neuro_cw) / var_demand
        bw_erp   = _variance(past_erp_cw) / var_demand

        # ── Build Week Payload ────────────────────────────────────────────────
        cross_node = self.info_bus.get_weekly_digest(self.week)

        payload = {
            "week":           self.week,
            "customer_demand": round(gross_demand, 0),
            "organic_demand":  round(organic_demand, 1),
            "promo_flag":      promo_flag,
            "macro":           macro,
            "kalman": {
                "organic":   round(organic_demand, 1),
                "trend":     round(kalman_out.get("trend", 0), 2),
                "promo_lift":round(promo_spike, 1),
                "confidence":round(kalman_out.get("confidence", 0.5), 3),
            },
            "return_wave":    return_wave,
            "lead_times":     {k: v.get("mean_days", 0) for k, v in lead_estimates.items()},
            "node_states":    node_states,
            "physics":        physics,
            "proposed_orders":proposed_orders,
            "erp_orders":     erp_orders,
            "final_orders":   final_orders,
            "guardrail_results": guardrail_results,
            "guardrail_log":  guardrail_log,
            "alerts":         alerts_dicts,
            "alert_summary":  self.alert_engine.get_dashboard_summary(),
            "cross_node_visibility": cross_node,
            "bullwhip": {
                "neuro_ratio":  round(bw_neuro, 3),
                "erp_ratio":    round(bw_erp, 3),
                "compression":  round((bw_erp - bw_neuro) / max(bw_erp, 0.001), 3),
            },
            "greige_stock_batches": self.greige_stock_batches,
            "season_weeks_remaining": season_weeks_remaining,
        }

        self.sim_log.append({
            "week":   self.week,
            "demand": gross_demand,
            "final_orders": final_orders,
            "erp_orders": erp_orders,
            "inventory": {n: self.nodes[n].inventory for n in self.NODES},
        })

        return payload

    def run_bulk(self, n_weeks: int = 12) -> list:
        return [self.step() for _ in range(min(n_weeks, 104 - self.week + 1))]

    def chat(self, node: str, message: str) -> dict:
        self._load_modules()
        if self.week == 0:
            return {"response": "Please run at least 1 week of simulation first.", "node": node, "week": 0, "fallback": True}

        node_state    = getattr(self, "latest_node_states", {}).get(node, {})
        macro         = getattr(self, "latest_macro", {"freight_congestion_index": 100})
        cross_node    = self.info_bus.get_cross_node_visibility(node)
        phys          = self.physics_engines[node].evaluate(node_state, macro)
        active_alerts = [
            {"id": a.id, "node": a.node, "severity": a.severity.value, "title": a.title,
             "detail": a.detail, "action": a.action, "week": a.week}
            for a in self.alert_engine.get_node_alerts(node)
        ]
        g_log = [g for g in (self.sim_log[-1].get("guardrail_log", []) if self.sim_log else []) if g.get("node") == node]

        return self._chat.send(
            node=node, message=message, node_state=node_state,
            cross_node_visibility=cross_node, physics=phys,
            active_alerts=active_alerts, guardrail_log=g_log,
            week=self.week,
        )

    def reset(self):
        self.__init__(self.sector)
