"""
information_bus.py — ChronoWeave Brain Layer
=============================================
Publish-subscribe nervous system for the 4-node Indian omnichannel
supply chain.  Solves the core ERP blindspot: every node now has full
chain transparency instead of seeing only its immediate downstream.

Nodes
-----
  store              — Node 1 : Retail Stores
  central_warehouse  — Node 2 : Central Warehouse
  mill               — Node 3 : Fabric Mill
  procurement        — Node 4 : Procurement / Pricing
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable


# ---------------------------------------------------------------------------
# NodeMessage — typed, prioritised envelope
# ---------------------------------------------------------------------------

class NodeMessage:
    """
    Typed message with full provenance metadata.

    Attributes
    ----------
    id          : unique "{sender_node}_{signal_type}_{week}" key
    sender_node : originating node
    signal_type : e.g. "DEMAND_SIGNAL", "RETURN_FORECAST"
    payload     : free-form dict with signal-specific fields
    week        : simulation week number
    priority    : "CRITICAL" | "HIGH" | "NORMAL" | "LOW"
    timestamp   : wall-clock time.time() at creation
    """

    PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}

    def __init__(
        self,
        sender_node: str,
        signal_type: str,
        payload: dict,
        week: int,
        priority: str = "NORMAL",
    ) -> None:
        self.id = f"{sender_node}_{signal_type}_{week}"
        self.sender_node = sender_node
        self.signal_type = signal_type
        self.payload = payload
        self.week = week
        self.priority = priority
        self.timestamp = time.time()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NodeMessage(id={self.id!r}, priority={self.priority!r}, "
            f"week={self.week})"
        )

    def __lt__(self, other: "NodeMessage") -> bool:
        return (
            self.PRIORITY_ORDER.get(self.priority, 99)
            < self.PRIORITY_ORDER.get(other.priority, 99)
        )


# ---------------------------------------------------------------------------
# InformationBus
# ---------------------------------------------------------------------------

class InformationBus:
    """
    Central nervous system — enables visibility-sharing across all supply
    chain nodes.

    Key capabilities
    ----------------
    * Publish / subscribe with typed callbacks.
    * 52-week rolling history per signal type.
    * Full cross-node visibility snapshot per requesting node.
    * Weekly digest of the entire chain.
    * Automated cascade-risk detection across 5 multi-node patterns.
    """

    # Signal catalogue — what each node is expected to broadcast
    NODE_SIGNALS: dict[str, list[str]] = {
        "store": [
            "NET_DEMAND",          # gross demand - returns in transit
            "SIZE_CURVE_HEALTH",   # per-size rack completeness
            "PROMO_FLAG",          # is this a promotional week?
            "STOCKOUT_SIZES",      # which sizes are stocked out
            "POS_VELOCITY",        # actual point-of-sale units/day
        ],
        "central_warehouse": [
            "RETURN_WAVE_FORECAST",  # predicted returns arriving in 7-14 days
            "EFFECTIVE_INVENTORY",   # on-hand + pipeline - predicted returns
            "PHANTOM_DEMAND_SPLIT",  # organic / promo / panic decomposition
            "FILL_RATE",             # % of store orders fulfilled
            "FACTORY_ORDER_SIGNAL",  # order placed to mill (post-guardrails)
        ],
        "mill": [
            "GREIGE_STOCK",         # undyed fabric available for conversion
            "PRODUCTION_QUEUE",     # current cut-make-trim schedule
            "SEASON_PROXIMITY",     # weeks to season end vs lead time
            "DYE_BATCH_STATUS",     # active dye batch capacity
            "LEAD_TIME_ESTIMATE",   # current estimated fulfillment time
        ],
        "procurement": [
            "COTTON_PRICE_SIGNAL",  # macro input: cotton index
            "YARN_HEDGE_STATUS",    # hedged vs spot procurement ratio
            "GST_CLIFF_STATUS",     # MRP proximity to ₹999/₹1000 threshold
            "SUPPLIER_LEAD_TIME",   # yarn delivery lead time
            "COST_PRESSURE_INDEX",  # composite input cost pressure
        ],
    }

    # Which signals each node is allowed to consume
    # (full transparency — every node sees the whole chain)
    _ALL_SIGNALS: list[str] = [
        sig
        for signals in NODE_SIGNALS.values()
        for sig in signals
    ]

    def __init__(self) -> None:
        # Ordered queue of in-flight messages (newest at end)
        self.message_queue: list[NodeMessage] = []

        # 52-week rolling history keyed by signal_type
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=52))

        # Subscriber callbacks keyed by signal_type
        self.subscribers: dict[str, list[Callable[[NodeMessage], None]]] = defaultdict(list)

        # Latest snapshot: signal_type → most-recent NodeMessage
        self.weekly_snapshot: dict[str, NodeMessage] = {}

    # -----------------------------------------------------------------------
    # Core pub/sub
    # -----------------------------------------------------------------------

    def publish(self, message: NodeMessage) -> None:
        """
        Publish a message onto the bus.

        * Appends to ``message_queue``.
        * Stores in rolling ``history[signal_type]``.
        * Updates ``weekly_snapshot``.
        * Fires all registered subscriber callbacks immediately.

        Parameters
        ----------
        message : NodeMessage to publish.
        """
        if not isinstance(message, NodeMessage):
            raise TypeError(f"Expected NodeMessage, got {type(message)}")

        self.message_queue.append(message)
        self.history[message.signal_type].append(message)
        self.weekly_snapshot[message.signal_type] = message

        # Notify subscribers
        for cb in self.subscribers.get(message.signal_type, []):
            try:
                cb(message)
            except Exception as exc:  # noqa: BLE001
                # Subscriber errors must never break the bus
                print(
                    f"[InformationBus] Subscriber callback error "
                    f"for {message.signal_type}: {exc}"
                )

    def subscribe(
        self,
        signal_type: str,
        callback: Callable[[NodeMessage], None],
    ) -> None:
        """
        Register *callback* to be invoked whenever *signal_type* is published.

        Parameters
        ----------
        signal_type : signal key to listen on (see NODE_SIGNALS).
        callback    : callable(NodeMessage) → None.
        """
        self.subscribers[signal_type].append(callback)

    # -----------------------------------------------------------------------
    # Cross-node visibility
    # -----------------------------------------------------------------------

    def get_cross_node_visibility(self, requesting_node: str) -> dict:
        """
        Return the full-chain state snapshot visible to *requesting_node*.

        This is the **core innovation** vs legacy ERP: instead of each node
        only seeing its immediate downstream, every node gets a live view of
        the entire supply chain.

        Returns
        -------
        dict with:
            requesting_node : str
            signals         : dict[signal_type → latest payload]
            by_node         : dict[node → dict[signal_type → latest payload]]
            snapshot_week   : most recent week seen on the bus
        """
        signals_out: dict[str, Any] = {}
        by_node: dict[str, dict[str, Any]] = {n: {} for n in self.NODE_SIGNALS}
        latest_week = 0

        for sig_type, msg in self.weekly_snapshot.items():
            signals_out[sig_type] = msg.payload
            if msg.sender_node in by_node:
                by_node[msg.sender_node][sig_type] = msg.payload
            latest_week = max(latest_week, msg.week)

        return {
            "requesting_node": requesting_node,
            "signals": signals_out,
            "by_node": by_node,
            "snapshot_week": latest_week,
        }

    # -----------------------------------------------------------------------
    # History access
    # -----------------------------------------------------------------------

    def get_signal_history(self, signal_type: str, weeks: int = 8) -> list[NodeMessage]:
        """
        Return the last *weeks* messages for *signal_type*, newest first.

        Parameters
        ----------
        signal_type : signal key.
        weeks       : number of records to return (capped at 52).
        """
        weeks = min(max(1, weeks), 52)
        hist = list(self.history[signal_type])
        return list(reversed(hist[-weeks:]))

    # -----------------------------------------------------------------------
    # Weekly digest
    # -----------------------------------------------------------------------

    def get_weekly_digest(self, week: int) -> dict:
        """
        Produce a full-chain digest for *week*.

        Collects the most recent payload for every signal across all nodes,
        labelled by node.  Useful for the dashboard header row.

        Returns
        -------
        dict with:
            week       : int
            nodes      : dict[node → dict[signal → payload or None]]
            guardrail_summary : count of distinct signal types seen this week
        """
        digest: dict[str, dict[str, Any]] = {n: {} for n in self.NODE_SIGNALS}

        for node, signals in self.NODE_SIGNALS.items():
            for sig in signals:
                msg = self.weekly_snapshot.get(sig)
                if msg is not None and msg.week == week:
                    digest[node][sig] = msg.payload
                else:
                    digest[node][sig] = None   # not yet published for this week

        return {
            "week": week,
            "nodes": digest,
            "guardrail_summary": {
                "signals_seen_this_week": sum(
                    1
                    for msg in self.weekly_snapshot.values()
                    if msg.week == week
                ),
                "total_signal_types": len(self._ALL_SIGNALS),
            },
        }

    # -----------------------------------------------------------------------
    # Cascade risk detection
    # -----------------------------------------------------------------------

    def detect_cascade_risk(self) -> list[dict]:
        """
        Cross-node pattern detection: are multiple nodes in stress simultaneously?

        Checks 5 specific multi-node risk patterns and returns a list of active
        risk dicts (empty list = no cascade risk detected).

        Risk Patterns
        -------------
        1. PHANTOM_DEMAND_CASCADE
           promo_flag=1  AND  return_wave_forecast > threshold
           → Promotional demand will create a phantom reorder surge when returns
             arrive; the bullwhip will be amplified by two signals at once.

        2. DEADSTOCK_CLIFF
           mill SEASON_PROXIMITY critical  AND  store NET_DEMAND high
           → Late-season fabric is being ordered to meet demand that can't be
             fulfilled before the season closes — potential greige deadstock.

        3. SIZE_CURVE_FILL_RATE_DOUBLE_MISS
           flagship ML completeness < 0.3  AND  central_warehouse fill_rate < 85 %
           → Flagship stores can't get M/L stock because the warehouse itself is
             short; lateral transfer won't solve it.

        4. COST_MARGIN_SQUEEZE_CLIFF
           cost_pressure_index > 1.15  AND  proposed_mrp near GST cliff
           → Rising input costs force MRP into the GST 12 % band, squeezing
             both consumer demand and margin simultaneously.

        5. RETURN_WAVE_INTAKE_OVERLOAD
           return_wave_forecast > fill_rate-adjusted intake capacity  AND
           warehouse effective_inventory already low
           → Returns will flood the warehouse at a time when it has no buffer
             to absorb them, causing throughput collapse.
        """
        risks: list[dict] = []
        snap = self.weekly_snapshot

        def _payload(signal: str) -> dict:
            msg = snap.get(signal)
            return msg.payload if msg else {}

        # -- helpers --
        promo_flag = int(_payload("PROMO_FLAG").get("active", 0))
        return_wave = float(_payload("RETURN_WAVE_FORECAST").get("predicted_units", 0))
        weekly_shipments = float(_payload("FACTORY_ORDER_SIGNAL").get("units_shipped", 1) or 1)
        return_rate = return_wave / weekly_shipments if weekly_shipments > 0 else 0.0

        season_proximity_critical = bool(
            _payload("SEASON_PROXIMITY").get("cliff_active", False)
        )
        net_demand = float(_payload("NET_DEMAND").get("units", 0))
        demand_high_threshold = float(_payload("NET_DEMAND").get("organic_baseline", net_demand or 1)) * 1.5

        flagship_ml = float(_payload("SIZE_CURVE_HEALTH").get("flagship_ML_completeness", 1.0))
        fill_rate = float(_payload("FILL_RATE").get("rate", 1.0))

        cost_pressure = float(_payload("COST_PRESSURE_INDEX").get("index", 1.0))
        proposed_mrp = int(_payload("GST_CLIFF_STATUS").get("proposed_mrp", 0))
        mrp_near_cliff = 990 <= proposed_mrp <= 1100

        effective_inv = float(_payload("EFFECTIVE_INVENTORY").get("units", 9999))
        intake_capacity = float(_payload("EFFECTIVE_INVENTORY").get("weekly_intake_capacity", 9999))

        # ---- Pattern 1: Phantom Demand Cascade ---------------------------
        if promo_flag and return_rate > 0.30:
            risks.append({
                "risk": "PHANTOM_DEMAND_CASCADE",
                "severity": "HIGH",
                "nodes_involved": ["store", "central_warehouse"],
                "detail": (
                    f"Promotional week active while return wave is {return_rate:.0%} "
                    "of shipments. Guardrail GR1 + GR2 under simultaneous pressure. "
                    "Phantom demand may inflate factory orders."
                ),
                "recommended_action": (
                    "Pre-arm GR1 return-wave netting. Freeze factory orders until "
                    "return wave settles. Validate organic demand baseline."
                ),
            })

        # ---- Pattern 2: Deadstock Cliff ----------------------------------
        if season_proximity_critical and net_demand > demand_high_threshold:
            risks.append({
                "risk": "DEADSTOCK_CLIFF",
                "severity": "CRITICAL",
                "nodes_involved": ["mill", "store"],
                "detail": (
                    f"Season cliff active (GR4 triggered) but store net demand "
                    f"({net_demand:.0f} units) is above 1.5× organic baseline. "
                    "Any new fabric ordered cannot reach stores before season end."
                ),
                "recommended_action": (
                    "Convert all remaining greige to bestseller SKUs immediately. "
                    "Accept markdown on slow-moving styles rather than ordering new fabric. "
                    "Escalate to Planning Head."
                ),
            })

        # ---- Pattern 3: Size Curve + Fill Rate Double Miss ---------------
        if flagship_ml < 0.3 and fill_rate < 0.85:
            risks.append({
                "risk": "SIZE_CURVE_FILL_RATE_DOUBLE_MISS",
                "severity": "HIGH",
                "nodes_involved": ["store", "central_warehouse"],
                "detail": (
                    f"Flagship M/L completeness is {flagship_ml:.0%} AND warehouse "
                    f"fill rate is {fill_rate:.0%}. Lateral transfer (GR3) cannot "
                    "compensate — the warehouse itself is undersupplied."
                ),
                "recommended_action": (
                    "Expedite M/L replenishment from mill on priority basis. "
                    "Temporarily reallocate M/L from B-grade stores to flagships. "
                    "Review WMS allocation rules."
                ),
            })

        # ---- Pattern 4: Cost-Margin Squeeze Cliff ------------------------
        if cost_pressure > 1.15 and mrp_near_cliff:
            risks.append({
                "risk": "COST_MARGIN_SQUEEZE_CLIFF",
                "severity": "CRITICAL",
                "nodes_involved": ["procurement", "store"],
                "detail": (
                    f"Cost pressure index {cost_pressure:.2f}x is forcing MRP "
                    f"(₹{proposed_mrp}) toward the ₹999/₹1001 GST cliff. "
                    "GR5 will lock MRP at ₹999; margin gap must be closed elsewhere."
                ),
                "recommended_action": (
                    "Accelerate hedged yarn lot procurement to offset spot-price surge. "
                    "Review pack-size mix for ₹799 bundle offer as alternative revenue path. "
                    "Flag for CFO sign-off on Q-end hedge strategy."
                ),
            })

        # ---- Pattern 5: Return Wave Intake Overload ----------------------
        if return_wave > intake_capacity * 0.9 and effective_inv < weekly_shipments * 1.5:
            risks.append({
                "risk": "RETURN_WAVE_INTAKE_OVERLOAD",
                "severity": "CRITICAL",
                "nodes_involved": ["central_warehouse"],
                "detail": (
                    f"Predicted return wave ({return_wave:.0f} units) is ≥ 90 % of "
                    f"warehouse intake capacity ({intake_capacity:.0f} units/week). "
                    f"Effective inventory buffer is only {effective_inv:.0f} units "
                    "(< 1.5 weeks cover)."
                ),
                "recommended_action": (
                    "Activate overflow returns dock at secondary 3PL. "
                    "Delay outbound shipments by 48 h to clear intake queue. "
                    "Alert returns QC team for triage priority processing."
                ),
            })

        return risks

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def flush_queue(self) -> list[NodeMessage]:
        """Drain and return all queued messages, sorted by priority."""
        out = sorted(self.message_queue)
        self.message_queue = []
        return out

    def node_health_check(self) -> dict[str, bool]:
        """
        Return True per node if it has published at least one signal this session.
        """
        seen_nodes: set[str] = {
            msg.sender_node for msg in self.weekly_snapshot.values()
        }
        return {node: node in seen_nodes for node in self.NODE_SIGNALS}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"InformationBus("
            f"queued={len(self.message_queue)}, "
            f"signal_types={len(self.weekly_snapshot)}, "
            f"subscribers={sum(len(v) for v in self.subscribers.values())})"
        )
