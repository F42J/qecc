# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: Route a fault-tolerant state preparation circuit onto a hardware graph.

This script demonstrates the full pipeline:
  1. Synthesize an FT state-prep circuit for the Steane [[7,1,3]] code.
  2. Build a square-lattice connectivity graph.
  3. Place and route the circuit using CNOT ladders.
  4. Verify fault-tolerance preservation via SparseVerificationSimulator.
"""

from __future__ import annotations

from math import ceil, sqrt

import networkx as nx
import numpy as np

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    CircuitLevelNoiseIdlingParallel,
    heuristic_prep_circuit,
    heuristic_verification_circuit,
)
from mqt.qecc.circuit_synthesis.ladder_place_and_route import LadderPlaceAndRoute
from mqt.qecc.circuit_synthesis.simulation import SparseVerificationSimulator

# ── 1. Synthesize the FT state-prep circuit ──────────────────────────────

code = CSSCode.from_code_name("steane")
prep = heuristic_prep_circuit(code)
ft_circuit = heuristic_verification_circuit(prep)

print(f"Input circuit: {ft_circuit.num_qubits} qubits, "
      f"depth {ft_circuit.depth()}, "
      f"{ft_circuit.count_ops().get('cx', 0)} CX gates")

# ── 2. Build a hardware connectivity graph ───────────────────────────────

# Size the grid to ~1.5x the number of logical qubits
grid_len = ceil(sqrt(ft_circuit.num_qubits * 1.5))
grid = nx.grid_2d_graph(grid_len, grid_len)

print(f"Hardware graph: {grid.number_of_nodes()} nodes, "
      f"{grid.number_of_edges()} edges "
      f"(fill rate: {ft_circuit.num_qubits / grid.number_of_nodes():.0%})")

# ── 3. Place and route ───────────────────────────────────────────────────

pnr = LadderPlaceAndRoute(grid, num_rounds=50, seed=42)
routed = pnr.schedule_circuit(ft_circuit)

print(f"\nRouted circuit: {routed.num_qubits} qubits, "
      f"depth {routed.depth()}, "
      f"{routed.count_ops().get('cx', 0)} CX gates")
print(f"Depth overhead: {routed.depth() / ft_circuit.depth():.1f}x")

# ── 4. Extract route qubit indices for the simulator ─────────────────────
#
# The routed circuit has three registers: "data", "ancilla", "route".
# The SparseVerificationSimulator needs to know which qubits are routing
# intermediates so it can exclude them from the FT verification.

route_indices = []
for qreg in routed.qregs:
    if qreg.name == "route":
        route_indices = [routed.find_bit(q).index for q in qreg]
        break

print(f"Route qubits: {len(route_indices)} "
      f"(excluded from FT verification)")

# ── 5. Verify fault-tolerance preservation ───────────────────────────────

sim = SparseVerificationSimulator(routed, code, route_indices)

ps = np.logspace(-4, -2, 5)
print(f"\n{'p':>10s}  {'p_L':>10s}")
print("-" * 24)

for p in ps:
    noise = CircuitLevelNoiseIdlingParallel(
        p_tqg=p, p_sqg=p, p_init=p, p_meas=p, p_idle=p / 100,
    )
    p_logical, _, _, _ = sim.logical_error_rate(noise, min_errors=10)
    print(f"{p:10.2e}  {p_logical:10.2e}")
