"""
glucostat_process.py

A simple, hackathon-friendly "glucostat" model implemented as a Process-bigraph Process.

What this file contains
-----------------------
1) GlucostatProcess
   - A Process-bigraph `Process` that simulates glucose regulation dynamics using:
       * A Bergman-style glucose/insulin core
       * A glucagon-based counter-regulatory pathway (stimulating hepatic glucose output)

2) A runnable test under:
       if __name__ == "__main__":

Model overview (high level)
---------------------------
We model a single "blood/plasma" compartment with four state variables:

  blood_glucose_mg_dL        (G):   blood glucose concentration
  plasma_insulin_uU_mL       (I):   plasma insulin concentration
  insulin_action_1_per_min   (X):   delayed/remote insulin effect (Bergman "remote compartment")
  plasma_glucagon_arb        (H):   plasma glucagon (arbitrary units; can be scaled later)

External driving inputs (rates) live in a dictionary called `inputs`:

  glucose_appearance_mg_dL_per_min    : glucose entering blood (e.g., meal absorption)
  exogenous_insulin_uU_mL_per_min     : insulin infused into plasma (e.g., pump)
  exogenous_glucagon_arb_per_min      : glucagon infused (e.g., rescue glucagon)

Core equations (conceptual)
---------------------------
- Glucose decreases via:
    (i) "glucose effectiveness" (insulin-independent clearance)
    (ii) insulin action (delayed insulin effect)

- Glucose increases via:
    (i) glucose appearance from a meal
    (ii) hepatic glucose output (stimulated by glucagon and suppressed by insulin)

- Insulin and glucagon are cleared toward baselines, and can be produced by simple
  threshold-linear endogenous secretion functions (enabled by gain parameters).

Process-bigraph conventions (matching the tutorials)
----------------------------------------------------
- inputs() returns schemas for state read ports
- outputs() returns schemas for state write ports
- update() returns DELTAS for the continuous state map (delta-merge behavior)
- time is written as overwrite[float] for a single authoritative clock

Requirements
------------
- SciPy (for odeint) is used in this version, following the tutorial_2 pattern.

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
    """Tutorial pattern: rebuild a Core that knows how to resolve local class addresses."""
    top = dict(inspect.getmembers(sys.modules["__main__"]))
    return allocate_core(top=top)


class GlucostatProcess(Process):
    """
    GlucostatProcess (Process-bigraph Process)

    This process simulates short-timescale glucose regulation using a compact ODE model.

    State (stored in the `state` map)
    --------------------------------
    state is a map[float] with the following keys:

      blood_glucose_mg_dL
          Blood glucose concentration (mg/dL).

      plasma_insulin_uU_mL
          Plasma insulin concentration (micro-units per mL).

      insulin_action_1_per_min
          "Remote" insulin action state (1/min). This represents delayed insulin effect
          on glucose uptake/clearance (Bergman minimal-model idea).

      plasma_glucagon_arb
          Plasma glucagon concentration in arbitrary units (can be scaled later).

    Inputs (stored in the `inputs` map)
    -----------------------------------
    inputs is a map[float] with the following keys:

      glucose_appearance_mg_dL_per_min
          Net glucose appearance into the blood compartment (mg/dL/min).
          Think: meal absorption already mapped into blood concentration units.

      exogenous_insulin_uU_mL_per_min
          Exogenous insulin infusion rate into plasma (uU/mL/min).

      exogenous_glucagon_arb_per_min
          Exogenous glucagon infusion rate (arb/min).

    Outputs
    -------
    - state: map[float]  (DELTA update: dy over the interval)
    - time_min: overwrite[float]  (absolute time, in minutes)

    Notes
    -----
    - Endogenous insulin/glucagon secretion are simple threshold-linear placeholders.
      Set secretion gains to 0.0 to disable them.
    - Hepatic glucose output is driven by glucagon and suppressed by insulin.
    - The intent is clarity + extensibility, not perfect physiology.
    """

    config_schema = {
        # --- Integration settings ---
        "internal_dt_min": {"_type": "float", "_default": 0.1},
        "odeint_kwargs": {"_type": "node", "_default": {}},

        # --- Baseline "set points" ---
        "baseline_glucose_mg_dL": {"_type": "float", "_default": 90.0},
        "baseline_insulin_uU_mL": {"_type": "float", "_default": 10.0},
        "baseline_glucagon_arb": {"_type": "float", "_default": 1.0},

        # --- Bergman-style glucose/insulin core parameters ---
        "glucose_effectiveness_1_per_min": {"_type": "float", "_default": 0.02},  # p1
        "insulin_action_decay_1_per_min": {"_type": "float", "_default": 0.03},   # p2
        "insulin_to_action_gain_1_per_min_per_uU_mL": {"_type": "float", "_default": 1e-4},  # p3

        # --- Insulin kinetics + simple endogenous secretion ---
        "insulin_clearance_1_per_min": {"_type": "float", "_default": 0.15},
        "insulin_secretion_gain_uU_mL_per_min_per_mg_dL": {"_type": "float", "_default": 0.0},
        "insulin_secretion_threshold_mg_dL": {"_type": "float", "_default": 90.0},

        # --- Glucagon kinetics + simple endogenous secretion ---
        "glucagon_clearance_1_per_min": {"_type": "float", "_default": 0.10},
        "glucagon_secretion_gain_arb_per_min_per_mg_dL": {"_type": "float", "_default": 0.05},
        "glucagon_secretion_threshold_mg_dL": {"_type": "float", "_default": 80.0},

        # --- Hepatic glucose output (H-driven, I-suppressed) ---
        "hepatic_glucose_output_basal_mg_dL_per_min": {"_type": "float", "_default": 0.5},
        "hepatic_glucose_output_gain_mg_dL_per_min_per_arb": {"_type": "float", "_default": 1.0},
        "hepatic_output_insulin_suppression_per_uU_mL": {"_type": "float", "_default": 0.02},

        # --- Safety / housekeeping ---
        "clamp_state_nonnegative": {"_type": "boolean", "_default": True},

        # Deterministic ordering (vectorize for odeint)
        "state_keys": {
            "_type": "list[string]",
            "_default": [
                "blood_glucose_mg_dL",
                "plasma_insulin_uU_mL",
                "insulin_action_1_per_min",
                "plasma_glucagon_arb",
            ],
        },
    }

    def initialize(self, config=None):
        # Basic validation
        keys = self.config.get("state_keys", [])
        if not isinstance(keys, (list, tuple)) or len(keys) == 0:
            raise ValueError("config['state_keys'] must be a non-empty list of strings")
        return self.config

    # -----------------------------
    # Process ports: schemas
    # -----------------------------
    def inputs(self):
        return {
            "state": "map[float]",
            "time_min": "float",
            "inputs": "map[float]",
        }

    def outputs(self):
        return {
            "state": "map[float]",           # DELTA update (tutorial style)
            "time_min": "overwrite[float]",  # authoritative clock update
        }

    # -----------------------------
    # Model RHS (dy/dt)
    # -----------------------------
    def _rhs_dict(self, state_map, t_min, inputs_map):
        """
        Compute time-derivatives of the model state.

        Parameters
        ----------
        state_map : dict
            Current state values (descriptive keys).
        t_min : float
            Current time in minutes (unused here, but included for generality).
        inputs_map : dict
            External driving inputs (rates).

        Returns
        -------
        dict
            Derivatives d(state)/dt keyed by the same state keys.
        """
        cfg = self.config

        # Unpack state (with sensible defaults)
        glucose = float(state_map.get("blood_glucose_mg_dL", cfg["baseline_glucose_mg_dL"]))
        insulin = float(state_map.get("plasma_insulin_uU_mL", cfg["baseline_insulin_uU_mL"]))
        insulin_action = float(state_map.get("insulin_action_1_per_min", 0.0))
        glucagon = float(state_map.get("plasma_glucagon_arb", cfg["baseline_glucagon_arb"]))

        # Unpack external inputs (rates)
        glucose_appearance = float(inputs_map.get("glucose_appearance_mg_dL_per_min", 0.0))
        exo_insulin = float(inputs_map.get("exogenous_insulin_uU_mL_per_min", 0.0))
        exo_glucagon = float(inputs_map.get("exogenous_glucagon_arb_per_min", 0.0))

        # --- Endogenous secretion (threshold-linear placeholders) ---
        # Insulin secretion activates above a glucose threshold.
        insulin_secretion = cfg["insulin_secretion_gain_uU_mL_per_min_per_mg_dL"] * max(
            0.0, glucose - cfg["insulin_secretion_threshold_mg_dL"]
        )

        # Glucagon secretion activates below a glucose threshold.
        glucagon_secretion = cfg["glucagon_secretion_gain_arb_per_min_per_mg_dL"] * max(
            0.0, cfg["glucagon_secretion_threshold_mg_dL"] - glucose
        )

        # --- Hepatic glucose output ---
        # Driven by glucagon above baseline, suppressed by insulin above baseline.
        insulin_suppression_factor = max(
            0.0,
            1.0 - cfg["hepatic_output_insulin_suppression_per_uU_mL"] * max(
                0.0, insulin - cfg["baseline_insulin_uU_mL"]
            ),
        )

        hepatic_output = (
            cfg["hepatic_glucose_output_basal_mg_dL_per_min"]
            + cfg["hepatic_glucose_output_gain_mg_dL_per_min_per_arb"]
            * max(0.0, glucagon - cfg["baseline_glucagon_arb"])
        ) * insulin_suppression_factor

        # --- Bergman-like core dynamics ---
        # Insulin action compartment: delayed insulin effect
        d_insulin_action = (
            -cfg["insulin_action_decay_1_per_min"] * insulin_action
            + cfg["insulin_to_action_gain_1_per_min_per_uU_mL"] * (insulin - cfg["baseline_insulin_uU_mL"])
        )

        # Glucose dynamics: clearance + appearance + hepatic output
        d_glucose = (
            -(cfg["glucose_effectiveness_1_per_min"] + insulin_action)
            * (glucose - cfg["baseline_glucose_mg_dL"])
            + glucose_appearance
            + hepatic_output
        )

        # Insulin dynamics: clearance toward baseline + secretion + infusion
        d_insulin = (
            -cfg["insulin_clearance_1_per_min"] * (insulin - cfg["baseline_insulin_uU_mL"])
            + insulin_secretion
            + exo_insulin
        )

        # Glucagon dynamics: clearance toward baseline + secretion + infusion
        d_glucagon = (
            -cfg["glucagon_clearance_1_per_min"] * (glucagon - cfg["baseline_glucagon_arb"])
            + glucagon_secretion
            + exo_glucagon
        )

        return {
            "blood_glucose_mg_dL": d_glucose,
            "plasma_insulin_uU_mL": d_insulin,
            "insulin_action_1_per_min": d_insulin_action,
            "plasma_glucagon_arb": d_glucagon,
        }

    # -----------------------------
    # Helpers: dict <-> vector
    # -----------------------------
    def _dict_to_vec(self, d, keys):
        return np.array([float(d.get(k, 0.0)) for k in keys], dtype=float)

    def _vec_to_dict(self, v, keys):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    # -----------------------------
    # Update step (odeint integration over interval)
    # -----------------------------
    def update(self, state, interval):
        """
        Advance the model state by `interval` minutes.

        The process returns a DELTA update for the `state` map, and overwrites `time_min`.

        Parameters
        ----------
        state : dict
            Current process input state, containing:
              state["state"]   -> map of state variables
              state["time_min"] -> current time in minutes
              state["inputs"]  -> map of external input rates
        interval : float
            Time step in minutes provided by the Composite scheduler.

        Returns
        -------
        dict
            {"state": delta_state_map, "time_min": new_time}
        """
        cfg = self.config
        keys = list(cfg["state_keys"])

        # Current values
        state_map_0 = dict(state.get("state", {}))
        time_0 = float(state.get("time_min", 0.0))
        inputs_map = dict(state.get("inputs", {}))

        # Fill missing state with defaults
        state_map_0.setdefault("blood_glucose_mg_dL", cfg["baseline_glucose_mg_dL"])
        state_map_0.setdefault("plasma_insulin_uU_mL", cfg["baseline_insulin_uU_mL"])
        state_map_0.setdefault("insulin_action_1_per_min", 0.0)
        state_map_0.setdefault("plasma_glucagon_arb", cfg["baseline_glucagon_arb"])

        # Integrate from time_0 to time_1 using odeint
        time_1 = time_0 + float(interval)
        internal_dt = float(cfg["internal_dt_min"])
        n_steps = max(2, int(np.ceil((time_1 - time_0) / internal_dt)) + 1)
        ts = np.linspace(time_0, time_1, n_steps)

        y0 = self._dict_to_vec(state_map_0, keys)

        def f(y_vec, t_min):
            y_dict = self._vec_to_dict(y_vec, keys)
            dy = self._rhs_dict(y_dict, t_min, inputs_map)
            return np.array([float(dy.get(k, 0.0)) for k in keys], dtype=float)

        odeint_kwargs = dict(cfg.get("odeint_kwargs", {}))
        traj = odeint(lambda yv, tt: f(yv, tt), y0, ts, **odeint_kwargs)
        y1 = traj[-1, :]

        # DELTA update (tutorial style)
        delta_vec = y1 - y0
        delta_map = self._vec_to_dict(delta_vec, keys)

        # Optional nonnegativity clamp by limiting the delta
        if cfg.get("clamp_state_nonnegative", True):
            for k in keys:
                if float(state_map_0.get(k, 0.0)) + float(delta_map[k]) < 0.0:
                    delta_map[k] = -float(state_map_0.get(k, 0.0))

        return {"state": delta_map, "time_min": float(time_1)}


def glucostat_initial_state(
    blood_glucose_mg_dL=90.0,
    plasma_insulin_uU_mL=10.0,
    insulin_action_1_per_min=0.0,
    plasma_glucagon_arb=1.0,
    time_min=0.0,
):
    """Convenience helper to build the expected state tree for this process."""
    return {
        "time_min": float(time_min),
        "state": {
            "blood_glucose_mg_dL": float(blood_glucose_mg_dL),
            "plasma_insulin_uU_mL": float(plasma_insulin_uU_mL),
            "insulin_action_1_per_min": float(insulin_action_1_per_min),
            "plasma_glucagon_arb": float(plasma_glucagon_arb),
        },
        "inputs": {
            "glucose_appearance_mg_dL_per_min": 0.0,
            "exogenous_insulin_uU_mL_per_min": 0.0,
            "exogenous_glucagon_arb_per_min": 0.0,
        },
    }


# -----------------------------
# Runnable test
# -----------------------------
if __name__ == "__main__":
    core = rebuild_core()
    print("✅ Core ready")

    PROC_ADDR = f"local:!{GlucostatProcess.__module__}.GlucostatProcess"
    print("Using process address:", PROC_ADDR)

    # Initial conditions
    init = glucostat_initial_state(
        blood_glucose_mg_dL=90.0,
        plasma_insulin_uU_mL=10.0,
        insulin_action_1_per_min=0.0,
        plasma_glucagon_arb=1.0,
        time_min=0.0,
    )

    # Turn on a simple endogenous insulin secretion rule so glucose responds to a "meal"
    process_config = {
        "internal_dt_min": 0.05,
        "odeint_kwargs": {"rtol": 1e-8, "atol": 1e-10},
        "insulin_secretion_gain_uU_mL_per_min_per_mg_dL": 0.05,
    }

    # Provide a constant "meal-like" glucose appearance drive for the whole simulation
    init["inputs"]["glucose_appearance_mg_dL_per_min"] = 2.0  # toy value

    sim = Composite(
        {
            "state": {
                # Shared state
                "time_min": init["time_min"],
                "state": init["state"],
                "inputs": init["inputs"],

                # Process node
                "glucostat": {
                    "_type": "process",
                    "address": PROC_ADDR,
                    "config": process_config,
                    "interval": 0.2,  # minutes per update
                    "inputs": {"time_min": ["time_min"], "state": ["state"], "inputs": ["inputs"]},
                    "outputs": {"time_min": ["time_min"], "state": ["state"]},
                },

                # Emitter to record time series
                "emitter": emitter_from_wires(
                    {
                        "global_time": ["global_time"],
                        "time_min": ["time_min"],
                        "blood_glucose_mg_dL": ["state", "blood_glucose_mg_dL"],
                        "plasma_insulin_uU_mL": ["state", "plasma_insulin_uU_mL"],
                        "insulin_action_1_per_min": ["state", "insulin_action_1_per_min"],
                        "plasma_glucagon_arb": ["state", "plasma_glucagon_arb"],
                        "glucose_appearance_mg_dL_per_min": ["inputs", "glucose_appearance_mg_dL_per_min"],
                        "exogenous_insulin_uU_mL_per_min": ["inputs", "exogenous_insulin_uU_mL_per_min"],
                        "exogenous_glucagon_arb_per_min": ["inputs", "exogenous_glucagon_arb_per_min"],
                    }
                ),
            }
        },
        core=core,
    )

    # Run for 10 minutes
    sim.run(10.0)

    records = sim.state["emitter"]["instance"].query()
    print("n records:", len(records))
    print("first record:", records[0])
    print("last record:", records[-1])

    # Basic sanity checks
    last = records[-1]
    for k in (
        "blood_glucose_mg_dL",
        "plasma_insulin_uU_mL",
        "insulin_action_1_per_min",
        "plasma_glucagon_arb",
    ):
        val = float(last[k])
        assert np.isfinite(val), f"{k} is not finite"
        assert val >= 0.0, f"{k} went negative"

    print("✅ GlucostatProcess test passed")