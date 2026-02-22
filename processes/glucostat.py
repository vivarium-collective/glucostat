"""
glucostat_process.py

Process-bigraph Process implementing a simple "glucostat" model:
  - Bergman minimal-model core (G, I, X)
  - Glucagon counter-regulation (H) driving hepatic glucose output
  - External interventions via input port u:
      meal_glucose_rate (mg/dL/min)
      insulin_infusion  (uU/mL/min)
      glucagon_infusion (arb/min)

This follows the tutorial style:
  - inputs:  y: map[float], t: float, u: map[float]
  - outputs: y: map[float] (DELTA update), t: overwrite[float]
  - uses scipy.integrate.odeint for integration
  - includes a runnable Composite test under __main__
"""

import sys
import inspect
import numpy as np

from process_bigraph import allocate_core
from process_bigraph.composite import Process, Composite
from process_bigraph.emitter import emitter_from_wires

try:
    from scipy.integrate import odeint
except Exception as e:
    raise ImportError(
        "This file requires SciPy for odeint.\n"
        "Install with: pip install scipy\n"
        f"Import error: {e}"
    )


def rebuild_core():
    """Tutorial pattern: rebuild core from __main__ symbol table."""
    top = dict(inspect.getmembers(sys.modules["__main__"]))
    return allocate_core(top=top)


class GlucostatProcess(Process):
    """
    Glucostat process in the same style as tutorial_2's ODEIntProcess.

    State variables in y:
      - G : blood glucose (mg/dL)
      - I : plasma insulin (uU/mL)
      - X : insulin action (1/min)   (remote compartment)
      - H : glucagon (arb)

    Inputs in u (rates):
      - meal_glucose_rate   (mg/dL/min)
      - insulin_infusion    (uU/mL/min)
      - glucagon_infusion   (arb/min)
    """

    config_schema = {
        # Integration
        "dt": {"_type": "float", "_default": 0.1},
        "odeint_kwargs": {"_type": "node", "_default": {}},

        # Baselines
        "G_basal": {"_type": "float", "_default": 90.0},
        "I_basal": {"_type": "float", "_default": 10.0},
        "H_basal": {"_type": "float", "_default": 1.0},

        # Bergman-ish parameters
        "p1": {"_type": "float", "_default": 0.02},   # 1/min glucose effectiveness
        "p2": {"_type": "float", "_default": 0.03},   # 1/min insulin action decay
        "p3": {"_type": "float", "_default": 1e-4},   # 1/(min*uU/mL) action gain

        # Insulin kinetics + simple endogenous secretion
        "kI": {"_type": "float", "_default": 0.15},   # 1/min clearance toward basal
        "insulin_secretion_gain": {"_type": "float", "_default": 0.0},
        "insulin_secretion_threshold": {"_type": "float", "_default": 90.0},

        # Glucagon kinetics + simple endogenous secretion
        "kH": {"_type": "float", "_default": 0.10},   # 1/min clearance toward basal
        "glucagon_secretion_gain": {"_type": "float", "_default": 0.05},
        "glucagon_secretion_threshold": {"_type": "float", "_default": 80.0},

        # Hepatic glucose output (H-driven, I-suppressed)
        "hepatic_output_basal": {"_type": "float", "_default": 0.5},   # mg/dL/min
        "hepatic_output_gain": {"_type": "float", "_default": 1.0},    # mg/dL/min per (H-H_basal)
        "hepatic_output_insulin_suppression": {"_type": "float", "_default": 0.02},  # per uU/mL above basal

        # Safety
        "clamp_nonnegative": {"_type": "boolean", "_default": True},

        # Deterministic ordering
        "state_keys": {"_type": "list[string]", "_default": ["G", "I", "X", "H"]},
    }

    def initialize(self, config=None):
        # Basic validation (tutorial style)
        keys = self.config.get("state_keys", ["G", "I", "X", "H"])
        if not isinstance(keys, (list, tuple)) or len(keys) == 0:
            raise ValueError("config['state_keys'] must be a non-empty list")
        return self.config

    def inputs(self):
        return {
            "y": "map[float]",
            "t": "float",
            "u": "map[float]",
        }

    def outputs(self):
        return {
            "y": "map[float]",          # DELTA
            "t": "overwrite[float]",    # clock overwrite
        }

    # -----------------------------
    # RHS
    # -----------------------------
    def _rhs_dict(self, y, t, u):
        cfg = self.config

        G = float(y.get("G", cfg["G_basal"]))
        I = float(y.get("I", cfg["I_basal"]))
        X = float(y.get("X", 0.0))
        H = float(y.get("H", cfg["H_basal"]))

        meal = float(u.get("meal_glucose_rate", 0.0))
        u_ins = float(u.get("insulin_infusion", 0.0))
        u_glu = float(u.get("glucagon_infusion", 0.0))

        # Simple endogenous secretion (threshold-linear placeholders)
        S_I = cfg["insulin_secretion_gain"] * max(0.0, G - cfg["insulin_secretion_threshold"])
        S_H = cfg["glucagon_secretion_gain"] * max(0.0, cfg["glucagon_secretion_threshold"] - G)

        # Hepatic output: basal + glucagon drive, suppressed by insulin above basal
        insulin_supp = max(
            0.0,
            1.0 - cfg["hepatic_output_insulin_suppression"] * max(0.0, I - cfg["I_basal"])
        )
        hepatic_output = (cfg["hepatic_output_basal"]
                          + cfg["hepatic_output_gain"] * max(0.0, H - cfg["H_basal"])) * insulin_supp

        # Bergman-like dynamics
        dX = -cfg["p2"] * X + cfg["p3"] * (I - cfg["I_basal"])
        dG = -(cfg["p1"] + X) * (G - cfg["G_basal"]) + meal + hepatic_output
        dI = -cfg["kI"] * (I - cfg["I_basal"]) + S_I + u_ins
        dH = -cfg["kH"] * (H - cfg["H_basal"]) + S_H + u_glu

        return {"G": dG, "I": dI, "X": dX, "H": dH}

    def _dict_to_vec(self, y_dict, keys):
        return np.array([float(y_dict.get(k, 0.0)) for k in keys], dtype=float)

    def _vec_to_dict(self, y_vec, keys):
        return {k: float(y_vec[i]) for i, k in enumerate(keys)}

    def update(self, state, interval):
        cfg = self.config
        keys = list(cfg.get("state_keys", ["G", "I", "X", "H"]))

        y0 = dict(state.get("y", {}))
        t0 = float(state.get("t", 0.0))
        u = dict(state.get("u", {}))

        # defaults if missing
        y0.setdefault("G", cfg["G_basal"])
        y0.setdefault("I", cfg["I_basal"])
        y0.setdefault("X", 0.0)
        y0.setdefault("H", cfg["H_basal"])

        t1 = t0 + float(interval)

        dt = float(cfg.get("dt", 0.1))
        odeint_kwargs = dict(cfg.get("odeint_kwargs", {}))

        n_steps = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
        ts = np.linspace(t0, t1, n_steps)

        y0_vec = self._dict_to_vec(y0, keys)

        def f(y_vec, t):
            y_dict = self._vec_to_dict(y_vec, keys)
            dy = self._rhs_dict(y_dict, t, u)
            return np.array([float(dy.get(k, 0.0)) for k in keys], dtype=float)

        traj = odeint(lambda yv, tt: f(yv, tt), y0_vec, ts, **odeint_kwargs)
        y1_vec = traj[-1, :]

        dy_vec = y1_vec - y0_vec
        dy = self._vec_to_dict(dy_vec, keys)

        if cfg.get("clamp_nonnegative", True):
            # prevent pushing state below zero via delta
            for i, k in enumerate(keys):
                if float(y0.get(k, 0.0)) + float(dy[k]) < 0.0:
                    dy[k] = -float(y0.get(k, 0.0))

        return {"y": dy, "t": float(t1)}


def glucostat_initial_state(G=90.0, I=10.0, X=0.0, H=1.0, t=0.0):
    return {
        "t": float(t),
        "y": {"G": float(G), "I": float(I), "X": float(X), "H": float(H)},
        "u": {
            "meal_glucose_rate": 0.0,
            "insulin_infusion": 0.0,
            "glucagon_infusion": 0.0,
        },
    }


# -----------------------------
# Runnable test
# -----------------------------
if __name__ == "__main__":
    core = rebuild_core()
    print("✅ Core ready")

    GLUCO_ADDR = f"local:!{GlucostatProcess.__module__}.GlucostatProcess"
    print("Using address:", GLUCO_ADDR)

    # Build a simple meal pulse: we'll keep u constant for each process interval.
    # For a quick test, we just set meal_glucose_rate to a positive value for the whole run.
    init = glucostat_initial_state(G=90, I=10, X=0, H=1, t=0)
    init["u"]["meal_glucose_rate"] = 2.0     # mg/dL/min glucose appearance (toy)
    init["u"]["insulin_infusion"] = 0.0
    init["u"]["glucagon_infusion"] = 0.0

    sim = Composite(
        {
            "state": {
                # shared state
                "t": init["t"],
                "y": init["y"],
                "u": init["u"],

                # process node
                "glucostat": {
                    "_type": "process",
                    "address": GLUCO_ADDR,
                    "config": {
                        "dt": 0.05,
                        "odeint_kwargs": {"rtol": 1e-8, "atol": 1e-10},
                        # keep default parameters, but you can tweak here
                        "insulin_secretion_gain": 0.05,  # turn on simple endogenous insulin secretion
                    },
                    "interval": 0.2,  # minutes per process update
                    "inputs": {"t": ["t"], "y": ["y"], "u": ["u"]},
                    "outputs": {"t": ["t"], "y": ["y"]},
                },

                # emitter time series
                "emitter": emitter_from_wires(
                    {
                        "time": ["global_time"],
                        "t": ["t"],
                        "G": ["y", "G"],
                        "I": ["y", "I"],
                        "X": ["y", "X"],
                        "H": ["y", "H"],
                        "meal": ["u", "meal_glucose_rate"],
                        "u_ins": ["u", "insulin_infusion"],
                        "u_glu": ["u", "glucagon_infusion"],
                    }
                ),
            }
        },
        core=core,
    )

    # Run for 10 minutes of composite time
    sim.run(10.0)

    records = sim.state["emitter"]["instance"].query()
    print("n records:", len(records))
    print("first record:", records[0])
    print("last record:", records[-1])

    # Basic sanity assertions
    last = records[-1]
    for k in ("G", "I", "X", "H"):
        val = float(last[k])
        assert np.isfinite(val), f"{k} is not finite"
        assert val >= 0.0, f"{k} went negative"

    print("✅ GlucostatProcess test passed")