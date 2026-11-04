"""
wardenclyffe_core.py — Wardenclyffe Thermodynamic Physics Engine (ChronoWeave port)
Provides adaptive snapshot timing and stress energy computation for the
Indian Omnichannel Apparel Supply Chain context.

Physics Interpretation:
  E_n  = System stress energy (backlog + stockout + return_wave + season_pressure)
  dt   = Adaptive decision granularity (how urgently should agents re-evaluate)
  η_s  = State-Space Filter learning rate (governs Kalman Q matrix scaling)
"""
import math
from collections import deque


class QuantumConstants:
    h_scaled   = 1e-3    # Scaled Planck constant
    gamma_hbar = 1.2     # γ × ℏ for adaptive learning rate
    gamma_star = 0.015   # Damping coefficient


class ThermodynamicController:
    def __init__(self, base_energy: float = 1.0):
        self.base_energy         = base_energy
        self.total_emergent_time = 0.0
        self.current_snapshot    = 0
        self.snapshot_history: deque = deque(maxlen=104)

    def calculate_energy(
        self,
        backlog:          float = 0.0,
        return_wave_risk: float = 0.0,
        size_curve_break: float = 0.0,   # fraction of M/L missing
        season_pressure:  float = 0.0,   # 0-1, high near EOSS
        freight_stress:   float = 0.0,
        promo_active:     bool  = False,
    ) -> float:
        promo_stress = 0.4 if promo_active else 0.0
        E_n = (
            self.base_energy
            + backlog          * 0.9
            + return_wave_risk * 0.7
            + size_curve_break * 0.6
            + season_pressure  * 1.2   # Season cliff is highest stress
            + freight_stress   * 0.3
            + promo_stress
        )
        return max(E_n, 1e-6)

    def tick_snapshot(self, E_n: float) -> float:
        dt = QuantumConstants.h_scaled / E_n
        self.total_emergent_time += dt
        self.current_snapshot    += 1
        self.snapshot_history.append({
            "snapshot": self.current_snapshot,
            "E_n": round(E_n, 5),
            "dt":  round(dt, 8),
            "t_emergent": round(self.total_emergent_time, 6),
        })
        return dt


class NodePhysicsEngine:
    """
    Per-node Wardenclyffe engine for ChronoWeave.
    Maps apparel supply chain stress to physics telemetry.
    """
    INFERENCE_MODES = {
        (0.0,  0.15): "EQUILIBRIUM",
        (0.15, 0.35): "CAUTIOUS_ADAPT",
        (0.35, 0.60): "ACTIVE_REBALANCE",
        (0.60, 0.85): "CRISIS_RESPONSE",
        (0.85, 9999): "SYSTEMIC_LOCK",
    }

    def __init__(self, node_id: str, base_demand: float = 2200.0):
        self.node_id     = node_id
        self.base_demand = base_demand
        self.clock       = ThermodynamicController(base_energy=base_demand / 100)

    def evaluate(self, node_state: dict, macro: dict) -> dict:
        backlog          = float(node_state.get("backlog_units", 0))
        return_wave      = float(node_state.get("predicted_returns_week", 0))
        ml_completeness  = float(node_state.get("ml_size_completeness", 1.0))
        season_weeks_rem = float(node_state.get("season_weeks_remaining", 26))
        freight_idx      = float(macro.get("freight_congestion_index", 100))
        promo            = bool(node_state.get("promo_flag", False))

        # Normalise stresses to [0, 1]
        return_risk      = min(return_wave / max(self.base_demand * 0.5, 1), 1.0)
        size_break       = max(0.0, 1.0 - ml_completeness)
        season_stress    = max(0.0, 1.0 - season_weeks_rem / 26.0)   # High near end of 26-wk season
        freight_stress   = max(0.0, (freight_idx - 100) / 100)

        E_n = self.clock.calculate_energy(
            backlog          = backlog / max(self.base_demand, 1),
            return_wave_risk = return_risk,
            size_curve_break = size_break,
            season_pressure  = season_stress,
            freight_stress   = freight_stress,
            promo_active     = promo,
        )
        dt   = self.clock.tick_snapshot(E_n)
        eta_s = max(0.05, min(0.80, QuantumConstants.gamma_hbar / E_n))

        e_norm = min(E_n / max(self.base_demand / 100, 1), 1.0)
        mode   = "EQUILIBRIUM"
        for (lo, hi), m in self.INFERENCE_MODES.items():
            if lo <= e_norm < hi:
                mode = m
                break

        return {
            "node_id":          self.node_id,
            "epistemic_energy": round(E_n, 4),
            "dt":               round(dt, 8),
            "eta_s":            round(eta_s, 4),
            "inference_mode":   mode,
            "snapshot":         self.clock.current_snapshot,
            "t_emergent":       round(self.clock.total_emergent_time, 6),
            "bisociation_active": eta_s >= 0.50,
            "stress_components": {
                "backlog":       round(backlog / max(self.base_demand, 1), 3),
                "return_wave":  round(return_risk, 3),
                "size_break":   round(size_break, 3),
                "season":       round(season_stress, 3),
                "freight":      round(freight_stress, 3),
                "promo":        0.4 if promo else 0.0,
            }
        }
