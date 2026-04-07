# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Place and route fault-tolerant state preparation circuits onto hardware connectivity graphs."""

from __future__ import annotations

__all__ = ["LadderPlaceAndRoute"]

import itertools
import logging
import operator
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse import diags as sparse_diags
from scipy.sparse.linalg import cg as sparse_cg
from scipy.spatial.distance import cdist

try:
    from ortools.sat.python import cp_model
except ImportError as _err:
    msg = "LadderPlaceAndRoute requires ortools. Install with: pip install mqt-qecc[routing]"
    raise ImportError(msg) from _err

logger = logging.getLogger(__name__)


class LadderPlaceAndRoute:
    """Place and route a fault-tolerant quantum circuit onto a physical connectivity graph.

    Uses CNOT ladders with post-ladder resets as the fault-tolerance-preserving
    routing primitive.  The default configuration uses wirelength-only placement
    (Approach A congestion forces disabled) with post-routing contention feedback
    (Approach B / CRISP, 2 iterations).

    To re-enable in-placement congestion forces (Approach A), set resist_k_load,
    resist_k_spike, resist_lambda_clear, and resist_alpha_occ to non-zero values.

    Pipeline:
      1. Build a dependency DAG and weighted interaction graph from the input circuit.
      2. Place logical qubits onto physical nodes via a resistance-based force-directed
         model (effective-resistance spring constants on the hardware graph Laplacian),
         refined by local swap search.
      3. Resolve macro-level ancilla conflicts with a CP-SAT pre-schedule that
         injects ordering edges into the DAG to prevent deadlocks.
      4. Route 2Q gates through CNOT-ladder paths in a sliding-window CP-SAT solver
         (NoOverlap on physical nodes, shortest-path candidates, ALAP reset scheduling).
      5. Refine the placement via contention feedback (CRISP): extract contention
         delays from the initial routing, augment the swap-search cost, and re-route.
      6. Emit the output circuit with post-ladder resets.
    """

    def __init__(
        self,
        connectivity: nx.Graph,
        # --- Circuit analysis ---
        alpha: float = 1.5,
        min_gate_weight: float = 0.1,
        # --- Placement strategy ---
        seed: int | None = None,
        placement_strategy: str = "post_selection",
        num_rounds: int = 5,
        verbose: bool = True,
        # --- Resistance-based placement: iteration control ---
        iterations: int = 100,
        dt: float = 0.1,
        # --- Resistance-based placement: physics parameters ---
        resist_damping: float = 0.85,
        resist_initial_temp: float = 1.0,
        resist_temp_decay: float = 0.95,
        resist_noise_scale: float = 0.1,
        resist_alpha_occ: float = 0.0,
        resist_lambda_clear: float = 0.0,
        resist_k_load: float = 0.0,
        resist_k_spike: float = 0.0,
        resist_r_protect: float = 0.5,
        resist_centering_strength: float = 0.01,
        resist_l_reg: float = 1e-4,
        resist_temporal_sigma: float = 0.15,
        resist_contrib_threshold: float = 0.01,
        resist_power_threshold_frac: float = 0.1,
        resist_r_eff_min: float = 0.1,
        resist_r_eff_same_node: float = 0.5,
        # --- Auto-calibrated ratios (multiplied by avg_edge_length) ---
        sigma_edge_ratio: float = 4.0,
        influence_edge_ratio: float = 6.0,
        k_repel_edge_coeff: float = 0.5,
        # --- Layout cost penalties ---
        layout_infeasible_penalty: float = 1_000_000,
        layout_superlinear_coeff: float = 20.0,
        layout_pessimism_infeasible: float = 200_000,
        layout_pessimism_detour: float = 50.0,
        # --- Placement refinement (local swap search) ---
        refine_pessimism: float = 0.15,
        # --- Sliding-window CP-SAT routing ---
        window_size: int = 30,
        commit_size: int = 15,
        path_limit: int = 3,
        # --- Spring exponent (1.0 = linear default, 2.0 = quadratic) ---
        spring_exponent: float = 1.0,
        spring_d_norm: float | None = None,
        # --- Contention-feedback refinement (CRISP-style) ---
        crisp_iterations: int = 2,
        crisp_swap_budget: int = 500,
        crisp_stall_limit: int = 100,
        crisp_penalty_scale: float = 1.0,
        # --- Computational effort (not model parameters) ---
        resist_temporal_bins: int = 10,
        resist_cg_maxiter: int = 200,
        resist_cg_rtol: float = 1e-6,
        resist_reff_refresh: int = 10,
        refine_swap_budget: int = 2000,
        refine_no_improvement_limit: int = 200,
        macro_solver_timeout: float = 5.0,
        cp_max_gate_duration: int = 20,
        cp_solver_timeout: float = 30.0,
        cp_num_workers: int = 1,
    ) -> None:
        """Initialize the place-and-route pipeline for the given connectivity graph."""
        self.phys_graph = connectivity
        self.alpha = alpha
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
        self.placement_strategy = placement_strategy
        self.num_rounds = num_rounds

        # Configure logging: if verbose and no handlers configured, add a basic one.
        # Users can override by configuring the logger for this module externally.
        if verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        elif not verbose and not logger.handlers:
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.WARNING)

        self.ancilla_nodes: set[str] = set()
        self.placement: dict[str, tuple] = {}

        self.iterations = iterations
        self.dt = dt

        self.resist_damping = resist_damping
        self.resist_initial_temp = resist_initial_temp
        self.resist_temp_decay = resist_temp_decay
        self.resist_noise_scale = resist_noise_scale
        self.resist_alpha_occ = resist_alpha_occ
        self.resist_k_repel = k_repel_edge_coeff  # will be scaled in _initialize_physical_board
        self.resist_lambda_clear = resist_lambda_clear
        self.resist_influence_radius = influence_edge_ratio  # will be scaled
        self.resist_k_load = resist_k_load
        self.resist_k_spike = resist_k_spike
        self.resist_r_protect = resist_r_protect
        self.resist_centering_strength = resist_centering_strength
        self.resist_l_reg = resist_l_reg
        self.resist_temporal_sigma = resist_temporal_sigma
        self.resist_temporal_bins = resist_temporal_bins
        self.resist_occupation_sigma = sigma_edge_ratio  # will be scaled
        self.resist_contrib_threshold = resist_contrib_threshold
        self.resist_power_threshold_frac = resist_power_threshold_frac
        self.resist_r_eff_min = resist_r_eff_min
        self.resist_r_eff_same_node = resist_r_eff_same_node
        self.resist_cg_maxiter = resist_cg_maxiter
        self.resist_cg_rtol = resist_cg_rtol

        self.resist_source_sigma = sigma_edge_ratio  # will be scaled
        self.resist_reff_refresh = resist_reff_refresh

        self.sigma_edge_ratio = sigma_edge_ratio
        self.influence_edge_ratio = influence_edge_ratio
        self.k_repel_edge_coeff = k_repel_edge_coeff

        self.min_gate_weight = min_gate_weight

        self.layout_infeasible_penalty = layout_infeasible_penalty
        self.layout_superlinear_coeff = layout_superlinear_coeff
        self.layout_pessimism_infeasible = layout_pessimism_infeasible
        self.layout_pessimism_detour = layout_pessimism_detour

        self.refine_pessimism = refine_pessimism
        self.refine_swap_budget = refine_swap_budget
        self.refine_no_improvement_limit = refine_no_improvement_limit

        self.macro_solver_timeout = macro_solver_timeout

        self.window_size = window_size
        self.commit_size = commit_size
        self.path_limit = path_limit
        self.cp_max_gate_duration = cp_max_gate_duration
        self.cp_solver_timeout = cp_solver_timeout
        self.cp_num_workers = cp_num_workers

        self.spring_exponent = spring_exponent
        self.spring_d_norm = spring_d_norm

        self.crisp_iterations = crisp_iterations
        self.crisp_swap_budget = crisp_swap_budget
        self.crisp_stall_limit = crisp_stall_limit
        self.crisp_penalty_scale = crisp_penalty_scale

        self._initialize_physical_board()

    def _initialize_physical_board(self) -> None:
        """Extract node list, coordinates, topology metrics, and auto-calibrate parameters."""
        self.phys_nodes = list(self.phys_graph.nodes())
        p = len(self.phys_nodes)

        # Canonical integer index for every physical node (enables deterministic
        # sorting of mixed-type node labels like tuples and strings).
        self.phys_node_to_idx = {n: i for i, n in enumerate(self.phys_nodes)}

        pos_dict = nx.kamada_kawai_layout(self.phys_graph)
        self.phys_coords = np.array([pos_dict[n] for n in self.phys_nodes])

        # --- Coordinate-based bounds (replaces grid_w / grid_h) ---
        self.coord_min = self.phys_coords.min(axis=0)
        self.coord_max = self.phys_coords.max(axis=0)

        # --- Topology metrics ---
        edge_lengths = np.array([
            np.linalg.norm(self.phys_coords[self.phys_node_to_idx[u]] - self.phys_coords[self.phys_node_to_idx[v]])
            for u, v in self.phys_graph.edges()
        ])
        self.avg_edge_length = float(np.mean(edge_lengths)) if len(edge_lengths) > 0 else 1.0

        degrees = np.array([self.phys_graph.degree(n) for n in self.phys_nodes], dtype=float)
        self.avg_degree = float(np.mean(degrees))
        self.max_degree = int(np.max(degrees))

        # --- Auto-calibrate sigma and influence radius ---
        # Target: sigma / avg_edge ≈ sigma_edge_ratio, matching the grid regime where
        # the Gaussian is selective (top-5 nodes capture ~25-30% of weight)
        # but smooth enough for gradient-based optimisation.
        auto_sigma = self.avg_edge_length * self.sigma_edge_ratio

        self.resist_source_sigma = auto_sigma
        self.resist_occupation_sigma = auto_sigma
        self.resist_influence_radius = self.avg_edge_length * self.influence_edge_ratio

        # Scale repulsion so that force magnitude at 1-edge distance is
        # comparable across topologies.
        self.resist_k_repel = self.k_repel_edge_coeff * (self.avg_edge_length**2)

        # Auto-scale CP-SAT gate duration to accommodate longer CNOT ladders.
        # A path of length d produces a ladder of duration 2*(d-1)+1.
        try:
            diameter = nx.diameter(self.phys_graph)
        except nx.NetworkXError:
            diameter = len(self.phys_nodes)  # disconnected fallback
        max_ladder_dur = 2 * (diameter - 1) + 1
        self.cp_max_gate_duration = max(self.cp_max_gate_duration, max_ladder_dur + 5)

        self._log(
            f"  [TOPOLOGY] {p} nodes, {self.phys_graph.number_of_edges()} edges, "
            f"avg_degree={self.avg_degree:.1f}, max_degree={self.max_degree}, diameter={diameter}"
        )
        self._log(
            f"  [TOPOLOGY] avg_edge_length={self.avg_edge_length:.4f}, "
            f"auto sigma={auto_sigma:.4f}, influence_r={self.resist_influence_radius:.4f}"
        )
        self._log(f"  [TOPOLOGY] cp_max_gate_duration={self.cp_max_gate_duration} (max_ladder={max_ladder_dur})")

    def _node_sort_key(self, node: object) -> int:
        """Canonical sort key for physical nodes of any type (tuples, strings, ints)."""
        return self.phys_node_to_idx.get(node, 0)

    @staticmethod
    def _log(msg: str, *args: object, level: int = logging.INFO, **kwargs: object) -> None:
        """Log a message via the module logger."""
        logger.log(level, msg, *args, **kwargs)

    @staticmethod
    def _debug(msg: str, *args: object, **kwargs: object) -> None:
        """Log a DEBUG-level message via the module logger."""
        logger.debug(msg, *args, **kwargs)

    # =========================================================================
    # 1. Circuit Analysis
    # =========================================================================

    def _build_dependency_graph(self, circuit: QuantumCircuit) -> tuple[nx.DiGraph, dict, list[dict]]:
        """Parse the circuit into a dependency DAG, qubit-pair map, and annotated gate list.

        Identifies ancilla qubits by register name and populates self.gate_locks,
        which maps each gate_id to the ancillas whose locked critical section
        begins at that gate (i.e. their first logical use in the circuit).
        """
        dep_graph = nx.DiGraph()
        qubit_last_gate = {}
        qubit_pair_nodes = {}
        gate_list = []
        self.ancilla_nodes = set()
        self.input_clbits = circuit.clbits

        first_ancilla_use = {}

        for i, instr in enumerate(circuit.data):
            name = instr.operation.name.lower()
            if name in {"barrier", "snapshot", "load", "save"}:
                continue

            q_args = instr.qubits
            c_args = instr.clbits

            q_names = []
            for q in q_args:
                bit_info = circuit.find_bit(q)
                reg, reg_idx = bit_info.registers[0]
                qn = f"{reg.name}_{reg_idx}"
                q_names.append(qn)
                if "anc" in qn.lower():
                    self.ancilla_nodes.add(qn)
                    if qn not in first_ancilla_use:
                        first_ancilla_use[qn] = i

            qubits_sorted = tuple(sorted(q_names))
            gate_node = f"gate_{i}"
            dep_graph.add_node(gate_node)

            for q in q_names:
                if q in qubit_last_gate:
                    dep_graph.add_edge(qubit_last_gate[q], gate_node)
                qubit_last_gate[q] = gate_node

            if len(q_args) == 2:
                qubit_pair_nodes.setdefault(qubits_sorted, []).append(gate_node)

            gate_info = {
                "id": gate_node,
                "idx": i,
                "qubits": q_names,
                "num_qubits": len(q_args),
                "instruction": instr.operation,
                "clbits": c_args,
            }
            if len(q_args) == 2:
                gate_info["source"] = q_names[0]
                gate_info["target"] = q_names[1]

            gate_list.append(gate_info)

        self.gate_locks = defaultdict(list)
        for anc, idx in first_ancilla_use.items():
            self.gate_locks[f"gate_{idx}"].append(anc)

        return dep_graph, qubit_pair_nodes, gate_list

    @staticmethod
    def _calculate_slack_and_depth(
        dep_graph: nx.DiGraph,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Compute ASAP depth and slack (= ALAP - ASAP) for every node in the DAG.

        Depth (ASAP): longest path from any source to this node.
        ALAP: latest start time without increasing the critical-path length.
        Slack: ALAP - depth.  Zero slack ↔ gate is on the critical path.
        """
        if not dep_graph.nodes():
            return {}, {}

        depth = {}
        for node in nx.topological_sort(dep_graph):
            preds = list(dep_graph.predecessors(node))
            depth[node] = (max(depth[p] for p in preds) + 1) if preds else 0

        max_depth = max(depth.values()) if depth else 0

        alap = {}
        for node in reversed(list(nx.topological_sort(dep_graph))):
            succs = list(dep_graph.successors(node))
            alap[node] = (min(alap[s] for s in succs) - 1) if succs else max_depth

        slack = {node: alap[node] - depth[node] for node in dep_graph.nodes()}

        return depth, slack

    def _create_weighted_graph(self, pair_nodes: dict, gate_list: list[dict]) -> nx.Graph:
        """Build a weighted logical interaction graph from qubit-pair gate statistics.

        Edge weight is the sum of pre-computed gate weights (from _compute_gate_weights)
        for all 2Q gates shared by each qubit pair.  This ensures the interaction graph
        uses the same weight formula (including the w_min floor) as the force model.
        """
        gate_weight_map = {g["id"]: g.get("weight", self.min_gate_weight) for g in gate_list}

        g = nx.Graph()
        for qubits, gate_ids in pair_nodes.items():
            u, v = qubits
            g.add_node(u)
            g.add_node(v)
            total_weight = sum(gate_weight_map.get(g, self.min_gate_weight) for g in gate_ids)
            if g.has_edge(u, v):
                g[u][v]["weight"] += total_weight
            else:
                g.add_edge(u, v, weight=total_weight)
        return g

    # =========================================================================
    # 2. Resistance-Based Placement
    # =========================================================================

    def _legalize_placement(self, continuous_pos: dict) -> dict:
        """Snap continuous 2D positions to physical nodes via minimum-cost assignment."""
        l_qs = list(continuous_pos.keys())
        if not l_qs:
            return {}
        l_c = np.array([continuous_pos[q] for q in l_qs])
        row, col = linear_sum_assignment(cdist(l_c, self.phys_coords))
        return {l_qs[r]: self.phys_nodes[c] for r, c in zip(row, col, strict=False)}

    def _get_maze_distance(self, start: object, end: object, occupied_blocking: set) -> int:
        """BFS shortest-path length avoiding occupied_blocking nodes."""
        if start == end:
            return 0
        queue, visited = [(start, 0)], {start}
        while queue:
            curr, d = queue.pop(0)
            if curr == end:
                return d
            for n in self.phys_graph.neighbors(curr):
                if n not in visited and (n not in occupied_blocking or n == end):
                    visited.add(n)
                    queue.append((n, d + 1))
        return -1

    def _calculate_layout_cost(
        self,
        mapping: dict,
        logical_g: nx.Graph,
        pessimism: float = 0.0,
        bottleneck_penalties: dict | None = None,
    ) -> float:
        """Evaluate placement quality as weighted sum of maze distances.

        Includes a super-linear penalty for long routes, an optional
        pessimistic term that penalises reliance on ancilla positions being
        free during routing, and optional per-node bottleneck penalties from
        contention-feedback analysis (CRISP-style).
        """
        cost = 0.0
        data_blocking = set()
        all_occupied = set()
        for log_q, pos in mapping.items():
            all_occupied.add(pos)
            if log_q not in self.ancilla_nodes:
                data_blocking.add(pos)

        for u, v, d in logical_g.edges(data=True):
            if u not in mapping or v not in mapping:
                continue
            pu, pv = mapping[u], mapping[v]
            wt = d.get("weight", 1.0)

            dist_opt = self._get_maze_distance(pu, pv, data_blocking)
            if dist_opt == -1:
                cost += self.layout_infeasible_penalty * wt
                continue

            c = dist_opt * wt
            if dist_opt > 1:
                c += ((dist_opt - 1) ** 2) * self.layout_superlinear_coeff * wt

            if pessimism > 0:
                pes_blocking = all_occupied - {pu, pv}
                dist_pes = self._get_maze_distance(pu, pv, pes_blocking)
                if dist_pes == -1:
                    c += pessimism * self.layout_pessimism_infeasible * wt
                elif dist_pes > dist_opt:
                    c += pessimism * (dist_pes - dist_opt) * self.layout_pessimism_detour * wt

            if bottleneck_penalties and nx.has_path(self.phys_graph, pu, pv):
                path = nx.shortest_path(self.phys_graph, pu, pv)
                for node in path[1:-1]:
                    if node in bottleneck_penalties:
                        c += bottleneck_penalties[node] * wt

            cost += c
        return cost

    def _refine_placement(
        self,
        mapping: dict,
        logical_g: nx.Graph,
        seed_rng: int,
        bottleneck_penalties: dict | None = None,
        swap_budget: int | None = None,
        stall_limit: int | None = None,
    ) -> tuple[dict, float]:
        """Local-search refinement via random pairwise physical-node swaps.

        Accepts improvements under a blended optimistic/pessimistic cost.

        If bottleneck_penalties is provided (from contention feedback), the
        cost function includes penalties for routing through bottleneck nodes.
        """
        if swap_budget is None:
            swap_budget = self.refine_swap_budget
        if stall_limit is None:
            stall_limit = self.refine_no_improvement_limit

        p2l = dict.fromkeys(self.phys_nodes)
        p2l.update({p: lq for lq, p in mapping.items()})
        curr_cost = self._calculate_layout_cost(mapping, logical_g, self.refine_pessimism, bottleneck_penalties)
        no_imp = 0
        candidates = list(self.phys_nodes)
        rng = np.random.RandomState(seed_rng)

        for _ in range(swap_budget):
            idx = rng.choice(len(candidates), 2, replace=False)
            pa, pb = candidates[idx[0]], candidates[idx[1]]
            la, lb = p2l[pa], p2l[pb]
            if la is None and lb is None:
                continue
            new_map = mapping.copy()
            if la:
                new_map[la] = pb
            if lb:
                new_map[lb] = pa
            new_cost = self._calculate_layout_cost(new_map, logical_g, self.refine_pessimism, bottleneck_penalties)
            if new_cost < curr_cost:
                curr_cost = new_cost
                mapping = new_map
                p2l[pa], p2l[pb] = lb, la
                no_imp = 0
            else:
                no_imp += 1
            if no_imp > stall_limit:
                break

        opt_cost = self._calculate_layout_cost(mapping, logical_g, 0.0)
        return mapping, opt_cost

    def _get_initialization(
        self,
        logical_g: nx.Graph,
        seed: int,
        strategy: str = "kamada_kawai",
        perturb_scale: float = 0.0,
    ) -> np.ndarray:
        """Produce initial continuous positions via Kamada-Kawai, spectral, or random layout."""
        scale = np.max(self.phys_coords) if len(self.phys_coords) > 0 else 1.0
        rng = np.random.RandomState(seed)
        nodes = list(logical_g.nodes())
        n = len(nodes)

        if strategy in {"kamada_kawai", "spectral"}:
            g_unweighted = nx.Graph()
            for u, v in logical_g.edges():
                g_unweighted.add_edge(u, v, weight=1.0)

            layout_fn = nx.kamada_kawai_layout if strategy == "kamada_kawai" else nx.spectral_layout
            try:
                pos_dict = layout_fn(g_unweighted, scale=scale)
                pos = np.array([pos_dict[node] for node in nodes])
            except (nx.NetworkXError, ValueError, np.linalg.LinAlgError):
                pos = (rng.rand(n, 2) * 2 - 1) * scale
        elif strategy == "random":
            pos = (rng.rand(n, 2) * 2 - 1) * scale
        else:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)

        if perturb_scale > 0:
            pos += rng.randn(n, 2) * (scale * perturb_scale)

        return pos

    @staticmethod
    def _compute_reachability(dep_graph: nx.DiGraph) -> set[tuple[str, str]]:
        """Transitive closure of the DAG as a set of (ancestor, descendant) pairs."""
        reachable = set()
        for node in dep_graph.nodes():
            visited = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                for succ in dep_graph.successors(curr):
                    if succ not in visited:
                        visited.add(succ)
                        reachable.add((node, succ))
                        stack.append(succ)
        return reachable

    def _compute_gate_weights(
        self, gate_list: list[dict], dep_graph: nx.DiGraph
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Assign per-gate weights using a bounded multiplicative formula.

        Uses alpha * (1 - norm_slack) * (1 + (1 - norm_depth))
        where the depth multiplier (2 - norm_depth) ranges in [1, 2].
        Modifies gate_list in place by adding a 'weight' field.
        """
        depths, slack = self._calculate_slack_and_depth(dep_graph)
        max_depth = max(depths.values()) if depths else 1
        max_slack = max(slack.values()) if slack else 1

        for gate in gate_list:
            d_norm = depths.get(gate["id"], 0) / (max_depth + 1)
            s_norm = slack.get(gate["id"], 0) / (max_slack + 1)
            gate["weight"] = max(self.min_gate_weight, self.alpha * (1.0 - s_norm) * (2.0 - d_norm))

        return depths, slack

    @staticmethod
    def _build_resistance_laplacian(
        conductances_arr: np.ndarray,
        _edges: list,
        edge_node_indices: list[tuple[int, int]],
        p: int,
    ) -> csr_matrix:
        """Assemble a sparse weighted graph Laplacian from per-edge conductances."""
        rows, cols, vals = [], [], []
        diag = np.zeros(p)

        for ei, (ia, ib) in enumerate(edge_node_indices):
            c = conductances_arr[ei]
            rows.extend([ia, ib])
            cols.extend([ib, ia])
            vals.extend([-c, -c])
            diag[ia] += c
            diag[ib] += c

        for i in range(p):
            rows.append(i)
            cols.append(i)
            vals.append(diag[i])

        return csr_matrix((vals, (rows, cols)), shape=(p, p))

    @staticmethod
    def _compute_soft_weights(pos: np.ndarray, phys_coords: np.ndarray, sigma: float) -> np.ndarray:
        """Compute normalized Gaussian affinity weights from continuous positions to physical nodes.

        For each logical qubit q at pos[q], returns a (n, P) matrix W where:
          W[q, j] = exp(-||pos[q] - phys[j]||^2 / (2*sigma^2)) / Z_q

        Each row sums to 1.  When a qubit sits exactly on a physical node the
        distribution collapses to a near-delta; when it is between nodes the
        weight spreads smoothly.
        """
        sq_dist = cdist(pos, phys_coords, "sqeuclidean")
        w_soft = np.exp(-sq_dist / (2.0 * sigma**2))
        row_sums = w_soft.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)  # prevent division by zero
        w_soft /= row_sums
        return w_soft

    def _run_resistance_placement(
        self,
        logical_g: nx.Graph,
        gate_list: list[dict],
        dep_graph: nx.DiGraph,
        depths: dict[str, int],
        seed_rng: int,
        initial_pos: np.ndarray | None = None,
    ) -> dict:
        """Resistance-based placement on an arbitrary physical topology.

        Uses a **soft occupation model**: each logical qubit's presence on
        the physical graph is represented as a normalised Gaussian distribution
        over physical nodes (instead of a hard snap to the nearest node).
        Current injection, effective resistance, and congestion are all
        computed from these distributed sources, yielding smooth forces.

        Forces applied each iteration:
          1a. Attraction: spring constant ~ dynamic soft R_eff x gate weight.
          1b. Edge congestion: continuous repulsion from edges with high
              pairwise-independent current overlap (temporal profile phi weighted).
          2.  Voltage-drop clearing: proportional repulsion of all nearby data
              qubits from high-dissipation physical nodes, weighted by each
              qubit's Gaussian proximity (log-compressed power).
          +   All-pairs repulsion, centering, and decaying Gaussian noise.

        The soft R_eff is recomputed every `resist_reff_refresh` iterations
        using distributed dipole sources on the occupation-damped Laplacian.
        """
        nodes = list(logical_g.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {nd: i for i, nd in enumerate(nodes)}
        phys_node_idx = {nd: i for i, nd in enumerate(self.phys_nodes)}
        p = len(self.phys_nodes)
        edges = list(self.phys_graph.edges())
        e = len(edges)
        edge_node_indices = [(phys_node_idx[a], phys_node_idx[b]) for a, b in edges]

        neighbors_of = [[] for _ in range(p)]
        for ei, (ia, ib) in enumerate(edge_node_indices):
            neighbors_of[ia].append((ib, ei))
            neighbors_of[ib].append((ia, ei))

        scale = np.max(self.phys_coords) if len(self.phys_coords) > 0 else 1.0
        margin = self.avg_edge_length * 2.0
        clip_lo = self.coord_min - margin
        clip_hi = self.coord_max + margin

        if initial_pos is not None:
            pos = np.array([initial_pos[i].copy() for i in range(n)])
        else:
            pos = (np.random.RandomState(seed_rng).rand(n, 2) * 2 - 1) * scale

        rng = np.random.RandomState(seed_rng)

        two_q_gates = [g for g in gate_list if g["num_qubits"] == 2]
        g_2q = len(two_q_gates)
        if g_2q == 0:
            return dict(zip(nodes, pos, strict=False))

        gate_weights = np.array([g["weight"] for g in two_q_gates])
        gate_src_nidx = np.array([node_idx[g["source"]] for g in two_q_gates])
        gate_tgt_nidx = np.array([node_idx[g["target"]] for g in two_q_gates])

        # Skip congestion computation when all Approach A parameters are zero
        approach_a_active = (
            self.resist_alpha_occ != 0
            or self.resist_k_load != 0
            or self.resist_k_spike != 0
            or self.resist_lambda_clear != 0
        )

        # Independence matrix and temporal profiles (Approach A only)
        if approach_a_active:
            reachability = self._compute_reachability(dep_graph)
            m = np.ones((g_2q, g_2q), dtype=np.float64)
            for i, gi in enumerate(two_q_gates):
                m[i, i] = 0.0
                for j in range(i + 1, g_2q):
                    gj = two_q_gates[j]
                    if (gi["id"], gj["id"]) in reachability or (
                        gj["id"],
                        gi["id"],
                    ) in reachability:
                        m[i, j] = 0.0
                        m[j, i] = 0.0

            max_depth = max(depths.values()) if depths else 1
            sigma = self.resist_temporal_sigma
            t = self.resist_temporal_bins
            time_bins = np.linspace(0, 1, t)
            phi = np.zeros((g_2q, t))
            for i, gate in enumerate(two_q_gates):
                d_norm = depths.get(gate["id"], 0) / max(max_depth, 1)
                phi[i, :] = np.exp(-0.5 * ((time_bins - d_norm) / sigma) ** 2)
        else:
            t = 1  # placeholder

        is_ancilla = np.array([nd in self.ancilla_nodes for nd in nodes])

        src_sigma = self.resist_source_sigma
        reff_refresh = self.resist_reff_refresh

        velocity = np.zeros((n, 2))
        damping = self.resist_damping
        temp = self.resist_initial_temp
        alpha_occ = self.resist_alpha_occ
        k_repel = self.resist_k_repel
        lambda_clear = self.resist_lambda_clear
        influence_radius = self.resist_influence_radius
        k_load = self.resist_k_load
        k_spike = self.resist_k_spike
        r_protect = self.resist_r_protect
        l_reg_diag = self.resist_l_reg
        occ_sigma = self.resist_occupation_sigma

        gate_r_eff = np.full(g_2q, self.resist_r_eff_min)
        last_reff_iteration = -reff_refresh  # force computation on iteration 0

        self._debug(f"    [RESISTANCE] {n} qubits, {g_2q} 2Q gates, {p} phys nodes, {e} edges")
        self._debug(f"    [RESISTANCE] Soft source sigma={src_sigma:.2f}, R_eff refresh every {reff_refresh} iters")

        for iteration in range(self.iterations):
            forces = np.zeros((n, 2))

            w_soft = self._compute_soft_weights(pos, self.phys_coords, src_sigma)

            if approach_a_active:
                data_pos = pos[~is_ancilla]
                occupation = (
                    np.sum(
                        np.exp(-cdist(data_pos, self.phys_coords, "sqeuclidean") / (2 * occ_sigma**2)),
                        axis=0,
                    )
                    if len(data_pos) > 0
                    else np.zeros(p)
                )
                node_damping = 1.0 / (1.0 + alpha_occ * occupation)
                cond = np.array([node_damping[ia] * node_damping[ib] for ia, ib in edge_node_indices])
            else:
                cond = np.ones(e)

            lap = self._build_resistance_laplacian(cond, edges, edge_node_indices, p)
            lap += sparse_diags([l_reg_diag], [0], shape=(p, p), format="csr")

            if iteration - last_reff_iteration >= reff_refresh:
                for gi in range(g_2q):
                    w_src = w_soft[gate_src_nidx[gi]]
                    w_tgt = w_soft[gate_tgt_nidx[gi]]
                    rhs = w_src - w_tgt
                    if np.max(np.abs(rhs)) < 1e-12:
                        gate_r_eff[gi] = self.resist_r_eff_same_node
                        continue
                    x, _ = sparse_cg(
                        lap,
                        rhs,
                        maxiter=self.resist_cg_maxiter,
                        rtol=self.resist_cg_rtol,
                    )
                    gate_r_eff[gi] = max(rhs @ x, self.resist_r_eff_min)
                last_reff_iteration = iteration

                if iteration == 0:
                    self._debug(
                        f"    [RESISTANCE] Initial soft R_eff: min={gate_r_eff.min():.2f} "
                        f"mean={gate_r_eff.mean():.2f} max={gate_r_eff.max():.2f}"
                    )

            if approach_a_active:
                u = np.zeros((e, g_2q))
                node_power = np.zeros(p)

                for gi in range(g_2q):
                    w_src = w_soft[gate_src_nidx[gi]]
                    w_tgt = w_soft[gate_tgt_nidx[gi]]
                    rhs = w_src - w_tgt
                    if np.max(np.abs(rhs)) < 1e-12:
                        continue

                    x, _ = sparse_cg(
                        lap,
                        rhs,
                        maxiter=self.resist_cg_maxiter,
                        rtol=self.resist_cg_rtol,
                    )
                    w = gate_weights[gi]

                    for ei, (ia, ib) in enumerate(edge_node_indices):
                        u[ei, gi] = cond[ei] * abs(x[ia] - x[ib]) * w

                    for j in range(p):
                        pj = sum(cond[ei] * (x[j] - x[nb_j]) ** 2 for nb_j, ei in neighbors_of[j])
                        node_power[j] += pj * w

                # Pairwise congestion: J(e,t) = diag(U*phi)^T M (U*phi)
                j_cong = np.zeros((e, t))
                for t_idx in range(t):
                    w_t = u * phi[:, t_idx]
                    j_cong[:, t_idx] = np.sum((w_t @ m) * w_t, axis=1)

                j_avg = np.mean(j_cong, axis=1)
                j_peak = np.max(j_cong, axis=1)
                global_avg = np.mean(j_avg) + 1e-12

            for gi in range(g_2q):
                r_eff = gate_r_eff[gi]
                w = gate_weights[gi]
                if r_eff < 1e-8:
                    continue
                ui, vi = gate_src_nidx[gi], gate_tgt_nidx[gi]
                delta = pos[vi] - pos[ui]
                # Quadratic spring: scale displacement by (dist / d_norm)^(exp-1)
                if abs(self.spring_exponent - 1.0) > 1e-12 and self.spring_d_norm is not None:
                    d = np.linalg.norm(delta)
                    if d > 1e-8:
                        delta *= (d / self.spring_d_norm) ** (self.spring_exponent - 1)
                attraction = delta * r_eff * w
                forces[ui] += attraction
                forces[vi] -= attraction

            # Force 1b: edge congestion repulsion (Approach A only)
            if approach_a_active:
                for ei in range(len(edges)):
                    r = j_avg[ei] / (global_avg + 1e-12)
                    s = j_peak[ei] / (j_avg[ei] + 1e-12)
                    f_mag = global_avg * (k_load * r * max(0.0, r - r_protect) + k_spike * r * max(0.0, s - 1.0))
                    if f_mag < 1e-6:
                        continue

                    ia, ib = edge_node_indices[ei]
                    mid = (self.phys_coords[ia] + self.phys_coords[ib]) / 2.0
                    for qi in range(n):
                        vec = pos[qi] - mid
                        d = np.linalg.norm(vec) + 1e-5
                        if d < influence_radius:
                            forces[qi] += (vec / d) * f_mag / (d + 0.5)

            # Force 2: voltage-drop clearing (Approach A only)
            if approach_a_active:
                power_threshold = self.resist_power_threshold_frac * (np.mean(node_power) + 1e-12)
                for j in range(p):
                    if node_power[j] < power_threshold:
                        continue
                    pn_coord = self.phys_coords[j]
                    clear_mag = lambda_clear * np.log1p(node_power[j])
                    for qi in range(n):
                        if is_ancilla[qi]:
                            continue
                        contrib = np.exp(-np.sum((pos[qi] - pn_coord) ** 2) / (2 * occ_sigma**2))
                        if contrib < self.resist_contrib_threshold:
                            continue
                        vec = pos[qi] - pn_coord
                        dist = np.linalg.norm(vec) + 1e-5
                        forces[qi] += (vec / dist) * clear_mag * contrib

            dmat = cdist(pos, pos) + 1e-5
            rmag = k_repel / (dmat**2)
            np.fill_diagonal(rmag, 0)
            dx = pos[:, 0][:, None] - pos[:, 0]
            dy = pos[:, 1][:, None] - pos[:, 1]
            forces[:, 0] += np.sum(dx * rmag, axis=1)
            forces[:, 1] += np.sum(dy * rmag, axis=1)

            centroid = np.mean(pos, axis=0)
            forces -= (pos - centroid) * self.resist_centering_strength

            if iteration % 25 == 0 or iteration == self.iterations - 1:
                f_mag = np.linalg.norm(forces, axis=1)
                top1_w = np.max(w_soft, axis=1).mean()
                top3_w = np.mean(np.sort(w_soft, axis=1)[:, -3:])
                diag_parts = (
                    f"    [ITER {iteration:3d}] |F|: mean={f_mag.mean():.4f} "
                    f"max={f_mag.max():.4f} | W: top1={top1_w:.3f} "
                    f"top3={top3_w:.3f} temp={temp:.4f}"
                )
                if approach_a_active:
                    cong_max = j_avg.max() if e > 0 else 0
                    power_max = node_power.max() if len(node_power) > 0 else 0
                    diag_parts += f" | cong={cong_max:.3f} power={power_max:.3f}"
                self._debug(diag_parts)

            velocity = damping * velocity + forces * self.dt
            velocity += rng.randn(n, 2) * temp * self.resist_noise_scale
            pos += velocity
            pos[:, 0] = np.clip(pos[:, 0], clip_lo[0], clip_hi[0])
            pos[:, 1] = np.clip(pos[:, 1], clip_lo[1], clip_hi[1])
            temp *= self.resist_temp_decay

        self._debug(
            f"    [RESISTANCE] Final soft R_eff: min={gate_r_eff.min():.2f} "
            f"mean={gate_r_eff.mean():.2f} max={gate_r_eff.max():.2f}"
        )
        self._debug("    [RESISTANCE] Placement complete")
        return dict(zip(nodes, pos, strict=False))

    def _print_placement_summary(self, placement: dict) -> None:
        """Print topology-agnostic placement diagnostics using graph distances."""
        p = len(self.phys_nodes)
        data_nodes_placed = [p for lq, p in placement.items() if lq not in self.ancilla_nodes]
        anc_nodes_placed = [p for lq, p in placement.items() if lq in self.ancilla_nodes]
        n_data, n_anc = len(data_nodes_placed), len(anc_nodes_placed)

        data_degrees = [self.phys_graph.degree(p) for p in data_nodes_placed]
        anc_degrees = [self.phys_graph.degree(p) for p in anc_nodes_placed]

        self._debug(
            f"    [PLACEMENT] {n_data}D + {n_anc}A on {p}-node graph "
            f"({len(placement)}/{p} used, {p - len(placement)} free)"
        )

        if data_degrees:
            deg_hist = Counter(data_degrees)
            deg_str = ", ".join(f"deg{d}:{c}" for d, c in sorted(deg_hist.items()))
            self._debug(
                f"    [PLACEMENT] Data node degrees: {deg_str} "
                f"(avg={np.mean(data_degrees):.1f}, topology max={self.max_degree})"
            )

        if anc_degrees:
            deg_hist = Counter(anc_degrees)
            deg_str = ", ".join(f"deg{d}:{c}" for d, c in sorted(deg_hist.items()))
            self._debug(f"    [PLACEMENT] Ancilla node degrees: {deg_str}")

        if n_data >= 2:
            dists = []
            dlist = data_nodes_placed
            for i in range(len(dlist)):
                for j in range(i + 1, len(dlist)):
                    if nx.has_path(self.phys_graph, dlist[i], dlist[j]):
                        dists.append(nx.shortest_path_length(self.phys_graph, dlist[i], dlist[j]))
                    else:
                        dists.append(-1)
            valid_dists = [d for d in dists if d >= 0]
            if valid_dists:
                self._debug(
                    f"    [PLACEMENT] Data-Data graph dist: "
                    f"min={min(valid_dists)} mean={np.mean(valid_dists):.1f} max={max(valid_dists)}"
                )

        if data_nodes_placed and anc_nodes_placed:
            dists = []
            for dp in data_nodes_placed:
                for ap in anc_nodes_placed:
                    if nx.has_path(self.phys_graph, dp, ap):
                        dists.append(nx.shortest_path_length(self.phys_graph, dp, ap))
                    else:
                        dists.append(-1)
            valid_dists = [d for d in dists if d >= 0]
            if valid_dists:
                self._debug(
                    f"    [PLACEMENT] Data-Anc graph dist: min={min(valid_dists)} mean={np.mean(valid_dists):.1f}"
                )

        occupied = set(placement.values())
        free_nodes = [n for n in self.phys_graph.nodes() if n not in occupied]
        free_with_neighbor = sum(1 for n in free_nodes if any(nb in occupied for nb in self.phys_graph.neighbors(n)))
        self._debug(f"    [PLACEMENT] Free nodes: {len(free_nodes)} ({free_with_neighbor} adjacent to qubits)")

        data_pos = {p for lq, p in placement.items() if lq not in self.ancilla_nodes}
        free_for_routing = [n for n in self.phys_graph.nodes() if n not in data_pos]
        g_free = self.phys_graph.subgraph(free_for_routing)
        comps = list(nx.connected_components(g_free))
        comp_sizes = sorted([len(c) for c in comps], reverse=True)
        if len(comps) > 1:
            self._debug(f"    [PLACEMENT] Free-space components: {len(comps)} (sizes: {comp_sizes[:5]})")
        else:
            self._debug(f"    [PLACEMENT] Free-space connected: 1 component ({comp_sizes[0]} nodes)")

        art_points = set(nx.articulation_points(g_free)) if g_free.number_of_nodes() > 1 else set()
        anc_on_art = sum(1 for lq, p in placement.items() if lq in self.ancilla_nodes and p in art_points)
        if anc_on_art > 0:
            self._debug(
                f"    [PLACEMENT] Ancillas on chokepoints: {anc_on_art}/{len(self.ancilla_nodes)} (locking risk)"
            )
        else:
            self._debug("    [PLACEMENT] No ancillas on chokepoints (lock-safe)")

    def run_placement_phase(
        self,
        circuit: QuantumCircuit,
        seed_val: int,
        strategy: str = "kamada_kawai",
        perturb: float = 0.0,
    ) -> tuple[dict, float]:
        """Full placement pipeline: analyse circuit -> resistance placement -> legalize -> refine."""
        dep_graph, pair_nodes, gate_list = self._build_dependency_graph(circuit)
        depth, _slack = self._calculate_slack_and_depth(dep_graph)

        self._compute_gate_weights(gate_list, dep_graph)
        logical_g_weighted = self._create_weighted_graph(pair_nodes, gate_list)

        # Store for CRISP contention-feedback loop reuse
        self._logical_g_weighted = logical_g_weighted

        initial_pos = self._get_initialization(logical_g_weighted, seed_val, strategy, perturb)
        continuous_pos = self._run_resistance_placement(
            logical_g_weighted, gate_list, dep_graph, depth, seed_val, initial_pos
        )

        rough_map = self._legalize_placement(continuous_pos)
        final_map, cost = self._refine_placement(rough_map, logical_g_weighted, seed_val)
        self._print_placement_summary(final_map)
        return final_map, cost

    # =========================================================================
    # 2.5. Macro Ancilla Scheduling (Deadlock Prevention)
    # =========================================================================

    def _analyze_ancilla_conflicts(self, gate_list: list[dict], dep_graph: nx.DiGraph, placement: dict) -> dict:
        """Detect physical path conflicts between ancilla critical sections.

        For each ancilla, determines its gate sequence, DAG-earliest start,
        and which other ancillas' physical locations are crossed by (k-1) or
        more of the k-shortest paths for each of its 2Q gates.
        """
        if not self.ancilla_nodes:
            return {}

        ancilla_locations = {anc: placement[anc] for anc in self.ancilla_nodes if anc in placement}

        ancilla_gates = {anc: [] for anc in self.ancilla_nodes}
        for gate in gate_list:
            for q in gate["qubits"]:
                if q in self.ancilla_nodes:
                    ancilla_gates[q].append(gate)

        depths = {}
        for node in nx.topological_sort(dep_graph):
            preds = list(dep_graph.predecessors(node))
            depths[node] = (max(depths[p] for p in preds) + 1) if preds else 0

        ancilla_info = {}
        # [DETERMINISM] Iterate ancillas in sorted order for deterministic dict construction
        for anc in sorted(self.ancilla_nodes):
            gates = ancilla_gates[anc]
            if not gates:
                continue

            gates_sorted = sorted(gates, key=operator.itemgetter("idx"))
            gate_ids = [g["id"] for g in gates_sorted]
            earliest_start = depths.get(gate_ids[0], 0) if gate_ids else 0

            uses = {}
            for gate_idx, gate in enumerate(gates_sorted):
                if gate["num_qubits"] != 2:
                    continue
                src_log = gate.get("source", gate["qubits"][0])
                tgt_log = gate.get("target", gate["qubits"][1])
                if src_log not in placement or tgt_log not in placement:
                    continue

                paths = self._get_k_shortest_pyramid_paths(placement[src_log], placement[tgt_log], placement)
                if not paths:
                    continue
                k = len(paths)

                # [DETERMINISM] Iterate ancilla_locations in sorted order
                for other_anc in sorted(ancilla_locations.keys()):
                    if other_anc == anc:
                        continue
                    other_loc = ancilla_locations[other_anc]
                    crossing_count = sum(1 for p in paths if other_loc in p)
                    if crossing_count >= k - 1 and crossing_count > 0:
                        if other_anc not in uses:
                            uses[other_anc] = [gate_idx, gate_idx]
                        else:
                            uses[other_anc][0] = min(uses[other_anc][0], gate_idx)
                            uses[other_anc][1] = max(uses[other_anc][1], gate_idx)

            ancilla_info[anc] = {
                "gates": gate_ids,
                "gates_sorted": gates_sorted,
                "duration": len(gate_ids),
                "earliest_start": earliest_start,
                "uses": uses,
            }

        return ancilla_info

    def _solve_macro_schedule(self, ancilla_info: dict, dep_graph: nx.DiGraph | None = None) -> dict:
        """CP-SAT macro scheduler: each ancilla is a single interval.

        Conflict constraints prevent overlapping usage of another ancilla's
        physical location during its critical section. Existing DAG dependencies
        between ancilla gates are extracted and enforced as ordering constraints.
        Objective: minimize makespan.

        Returns ancilla->start_time dict, or None if infeasible.
        """
        if not ancilla_info:
            return {}

        has_conflicts = any(info["uses"] for info in ancilla_info.values())
        if not has_conflicts:
            return {anc: info["earliest_start"] for anc, info in ancilla_info.items()}

        model = cp_model.CpModel()

        total_duration = sum(info["duration"] for info in ancilla_info.values())
        max_earliest = max(info["earliest_start"] for info in ancilla_info.values())
        horizon = total_duration + max_earliest + 100

        # [DETERMINISM] Iterate in sorted key order for deterministic variable creation
        start_vars = {}
        for anc in sorted(ancilla_info.keys()):
            info = ancilla_info[anc]
            start_vars[anc] = model.NewIntVar(0, horizon, f"start_{anc}")
            model.Add(start_vars[anc] >= info["earliest_start"])

        gate_to_anc_idx = {}
        for anc in sorted(ancilla_info.keys()):
            info = ancilla_info[anc]
            for idx, gate_id in enumerate(info["gates"]):
                gate_to_anc_idx[gate_id] = (anc, idx)

        dag_constraints_added = 0
        if dep_graph is not None:
            for u, v in dep_graph.edges():
                if u in gate_to_anc_idx and v in gate_to_anc_idx:
                    anc_u, idx_u = gate_to_anc_idx[u]
                    anc_v, idx_v = gate_to_anc_idx[v]
                    if anc_u != anc_v:
                        model.Add(start_vars[anc_u] + idx_u + 1 <= start_vars[anc_v] + idx_v)
                        dag_constraints_added += 1

        if dag_constraints_added > 0:
            self._log(f"    [MACRO] Added {dag_constraints_added} DAG dependency constraints")

        processed_pairs = set()
        # [DETERMINISM] Iterate in sorted order for deterministic constraint creation
        for anc_a in sorted(ancilla_info.keys()):
            info_a = ancilla_info[anc_a]
            for anc_b in sorted(info_a["uses"].keys()):
                if anc_b not in ancilla_info:
                    continue
                pair = tuple(sorted([anc_a, anc_b]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                info_b = ancilla_info[anc_b]
                a_uses_b = info_a["uses"].get(anc_b)
                b_uses_a = info_b["uses"].get(anc_a)

                a_before_b = model.NewBoolVar(f"{anc_a}_before_{anc_b}")

                if a_uses_b and b_uses_a:
                    model.Add(start_vars[anc_b] >= start_vars[anc_a] + a_uses_b[1] + 1).OnlyEnforceIf(a_before_b)
                    model.Add(start_vars[anc_a] >= start_vars[anc_b] + b_uses_a[1] + 1).OnlyEnforceIf(a_before_b.Not())
                elif a_uses_b:
                    model.Add(start_vars[anc_b] >= start_vars[anc_a] + a_uses_b[1] + 1).OnlyEnforceIf(a_before_b)
                    model.Add(start_vars[anc_a] + a_uses_b[0] >= start_vars[anc_b] + info_b["duration"]).OnlyEnforceIf(
                        a_before_b.Not()
                    )
                elif b_uses_a:
                    model.Add(start_vars[anc_b] + b_uses_a[0] >= start_vars[anc_a] + info_a["duration"]).OnlyEnforceIf(
                        a_before_b
                    )
                    model.Add(start_vars[anc_a] >= start_vars[anc_b] + b_uses_a[1] + 1).OnlyEnforceIf(a_before_b.Not())

        # [DETERMINISM] Build end_times in sorted order
        end_times = [start_vars[anc] + ancilla_info[anc]["duration"] for anc in sorted(ancilla_info.keys())]
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, end_times)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.macro_solver_timeout
        # [DETERMINISM] Default: single worker for reproducibility
        solver.parameters.num_search_workers = self.cp_num_workers
        solver.parameters.random_seed = self.seed % (2**31)
        status = solver.Solve(model)

        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            result = {anc: solver.Value(start_vars[anc]) for anc in ancilla_info}
            starts = sorted(result.values())
            self._log(
                f"    [MACRO] Solved: {len(result)} ancillas, "
                f"span={max(starts) - min(starts)}, range=[{min(starts)},{max(starts)}]"
            )
            return result

        self._log("    [MACRO] Schedule INFEASIBLE - placement has unresolvable ancilla conflicts")
        return None

    def _inject_cycle_breaking_edges(
        self, dep_graph: nx.DiGraph, ancilla_info: dict, macro_schedule: dict
    ) -> nx.DiGraph:
        """Add synthetic DAG edges that enforce the macro schedule ordering.

        For each conflict pair (A, B) where A is scheduled first: add an edge
        from A's last conflicting gate to B's first gate so B cannot start
        until A has cleared the conflict window.
        """
        if not macro_schedule or not ancilla_info:
            return dep_graph

        edges_added = 0
        for anc_a, info_a in ancilla_info.items():
            start_a = macro_schedule.get(anc_a, 0)
            for anc_b, conflict_window in info_a["uses"].items():
                if anc_b not in ancilla_info or anc_b not in macro_schedule:
                    continue
                start_b = macro_schedule.get(anc_b, 0)
                info_b = ancilla_info[anc_b]

                if start_a + conflict_window[1] < start_b and conflict_window[1] < len(info_a["gates"]):
                    release_gate = info_a["gates"][conflict_window[1]]
                    waiting_gate = info_b["gates"][0]
                    if not dep_graph.has_edge(release_gate, waiting_gate):
                        dep_graph.add_edge(release_gate, waiting_gate, synthetic=True, cycle_break=True)
                        edges_added += 1

        if edges_added > 0:
            self._log(f"    [CYCLE BREAK] Added {edges_added} dependency edges")
        return dep_graph

    # =========================================================================
    # 3. Sliding-Window CP-SAT Routing
    # =========================================================================

    def _diagnose_no_path(
        self,
        src: object,
        tgt: object,
        placement: dict,
        locked_ancillas: set | None = None,
    ) -> str:
        """Generate a diagnostic string explaining why no path exists between src and tgt."""
        data_obstacles = set()
        locked_anc_obstacles = set()
        for log_q, phys_n in placement.items():
            if phys_n in {src, tgt}:
                continue
            if log_q in self.ancilla_nodes:
                if locked_ancillas and log_q in locked_ancillas:
                    locked_anc_obstacles.add(phys_n)
            else:
                data_obstacles.add(phys_n)

        all_obstacles = data_obstacles | locked_anc_obstacles

        if not nx.has_path(self.phys_graph, src, tgt):
            return "DISCONNECTED in raw graph"
        raw_dist = nx.shortest_path_length(self.phys_graph, src, tgt)

        if nx.has_path(self.phys_graph, src, tgt):
            raw_path = nx.shortest_path(self.phys_graph, src, tgt)
            blocked_by_data = sum(1 for n in raw_path[1:-1] if n in data_obstacles)
            blocked_by_anc = sum(1 for n in raw_path[1:-1] if n in locked_anc_obstacles)
        else:
            blocked_by_data = blocked_by_anc = 0

        valid_nodes = [n for n in self.phys_graph.nodes() if n not in all_obstacles]
        g_sub = self.phys_graph.subgraph(valid_nodes)
        src_comp_size = len(nx.node_connected_component(g_sub, src)) if src in g_sub else 0
        tgt_comp_size = len(nx.node_connected_component(g_sub, tgt)) if tgt in g_sub else 0
        same_comp = tgt in nx.node_connected_component(g_sub, src) if src in g_sub and tgt in g_sub else False

        src_deg = self.phys_graph.degree(src)
        tgt_deg = self.phys_graph.degree(tgt)

        return (
            f"raw_dist={raw_dist}, shortest_blocked={blocked_by_data}D+{blocked_by_anc}A, "
            f"src_comp={src_comp_size}, tgt_comp={tgt_comp_size}, connected={same_comp}, "
            f"src_deg={src_deg}, tgt_deg={tgt_deg}, "
            f"obstacles={len(data_obstacles)}D+{len(locked_anc_obstacles)}A/{len(self.phys_graph.nodes())}"
        )

    def _get_k_shortest_pyramid_paths(
        self,
        src: object,
        tgt: object,
        placement: dict,
        locked_ancillas: set | None = None,
    ) -> list[list]:
        """Find up to path_limit shortest paths avoiding data qubits and locked ancillas."""
        if src == tgt:
            return []
        obstacles = set()
        for log_q, phys_n in placement.items():
            if log_q in self.ancilla_nodes:
                if locked_ancillas and log_q in locked_ancillas and phys_n not in {src, tgt}:
                    obstacles.add(phys_n)
            elif phys_n not in {src, tgt}:
                obstacles.add(phys_n)

        valid_nodes = [n for n in self.phys_graph.nodes() if n not in obstacles]
        g_sub = self.phys_graph.subgraph(valid_nodes)
        try:
            return list(itertools.islice(nx.shortest_simple_paths(g_sub, src, tgt), self.path_limit))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _solve_cp_window(
        self,
        gate_window: list,
        placement: dict,
        dep_graph: nx.DiGraph,
        completed_gates: dict,
        completed_resource_usage: dict,
        qubit_available_time: dict,
        locked_ancillas: set | None = None,
    ) -> dict | None:
        """Solve a single scheduling window with CP-SAT.

        Models each 2Q gate as a CNOT-ladder (forward + backward pass) with
        per-node interval variables and NoOverlap constraints. 1Q gates occupy
        a single node for 1-2 timesteps depending on reset/measurement needs.

        Objective hierarchy (lexicographic via weighted sum):
          1. Minimize makespan.
          2. Maximize reset start times (ALAP locking to keep corridors open).

        Returns gate_id->result dict, or None if any gate fails to route.
        """
        model = cp_model.CpModel()

        effective_locked = set(locked_ancillas) if locked_ancillas else set()
        gate_vars = {}
        node_intervals = defaultdict(list)
        failed_gates = []

        num_gates = len(gate_window)
        horizon = num_gates * self.cp_max_gate_duration + 100

        # [DETERMINISM] Sort gate_window by gate index for deterministic variable creation order
        sorted_gate_window = sorted(gate_window, key=operator.itemgetter("idx"))

        for gate in sorted_gate_window:
            g_id = gate["id"]
            starts_lock = g_id in self.gate_locks

            if gate["num_qubits"] != 2:
                # --- 1Q gate ---
                log_q = gate["qubits"][0]
                phys_node = placement[log_q]
                is_measure = gate["instruction"].name.lower() == "measure"

                gate_duration = 2 if is_measure else 1

                start_var = model.NewIntVar(0, horizon, f"start_{g_id}")
                end_var = model.NewIntVar(0, horizon, f"end_{g_id}")
                model.Add(end_var == start_var + gate_duration)
                interval = model.NewFixedSizeIntervalVar(start_var, gate_duration, f"usage_{g_id}")
                node_intervals[phys_node].append(interval)

                gate_vars[g_id] = {
                    "start": start_var,
                    "end": end_var,
                    "type": "1q",
                    "u_phys": phys_node,
                    "op": gate["instruction"],
                    "clbits": gate["clbits"],
                    "starts_lock": starts_lock,
                    "duration": gate_duration,
                    "is_measure": is_measure,
                }

                if phys_node in completed_resource_usage:
                    for comp_start, comp_end, comp_gate_id in completed_resource_usage[phys_node]:
                        b = model.NewBoolVar(f"no_overlap_{g_id}_{comp_gate_id}")
                        model.Add(end_var <= comp_start).OnlyEnforceIf(b)
                        model.Add(start_var >= comp_end).OnlyEnforceIf(b.Not())
            else:
                # --- 2Q gate (CNOT ladder) ---
                src_logical = gate.get("source", gate["qubits"][0])
                tgt_logical = gate.get("target", gate["qubits"][1])
                src_phys = placement[src_logical]
                tgt_phys = placement[tgt_logical]

                paths = self._get_k_shortest_pyramid_paths(src_phys, tgt_phys, placement, effective_locked)

                if not paths:
                    diag = self._diagnose_no_path(src_phys, tgt_phys, placement, effective_locked)
                    failed_gates.append({
                        "id": g_id,
                        "qubits": gate["qubits"],
                        "reason": f"No path from {src_phys} to {tgt_phys}",
                    })
                    self._log(f"  [ROUTING FAIL] {g_id}: {src_logical}{src_phys}->{tgt_logical}{tgt_phys} | {diag}")
                    continue

                start_var = model.NewIntVar(0, horizon, f"start_{g_id}")
                path_idx = model.NewIntVar(0, len(paths) - 1, f"path_{g_id}")

                durations = [2 * (len(p) - 2) + 1 for p in paths]

                duration_var = model.NewIntVar(min(durations), max(durations), f"dur_{g_id}")
                model.AddElement(path_idx, durations, duration_var)
                end_var = model.NewIntVar(0, horizon, f"end_{g_id}")
                model.Add(end_var == start_var + duration_var)

                gate_vars[g_id] = {
                    "start": start_var,
                    "end": end_var,
                    "path_idx": path_idx,
                    "paths": paths,
                    "type": "2q",
                    "source": src_logical,
                    "target": tgt_logical,
                    "starts_lock": starts_lock,
                    "durations": durations,
                }

                for k, path in enumerate(paths):
                    dist = len(path) - 1
                    is_selected = model.NewBoolVar(f"{g_id}_path_{k}")
                    model.Add(path_idx == k).OnlyEnforceIf(is_selected)
                    model.Add(path_idx != k).OnlyEnforceIf(is_selected.Not())

                    # Source interval
                    src_dur = durations[k]
                    src_end_calc = model.NewIntVar(0, horizon, f"src_e_{g_id}_{k}")
                    model.Add(src_end_calc == start_var + src_dur)
                    node_intervals[src_phys].append(
                        model.NewOptionalIntervalVar(
                            start_var,
                            src_dur,
                            src_end_calc,
                            is_selected,
                            f"src_{g_id}_{k}",
                        )
                    )

                    if src_phys in completed_resource_usage:
                        for (
                            comp_start,
                            comp_end,
                            comp_gate_id,
                        ) in completed_resource_usage[src_phys]:
                            b = model.NewBoolVar(f"no_overlap_src_{g_id}_{k}_{comp_gate_id}")
                            model.Add(src_end_calc <= comp_start).OnlyEnforceIf([is_selected, b])
                            model.Add(start_var >= comp_end).OnlyEnforceIf([is_selected, b.Not()])

                    # Peak (target node at ladder apex)
                    peak_start = model.NewIntVar(0, horizon, f"peak_s_{g_id}_{k}")
                    model.Add(peak_start == start_var + (dist - 1))

                    node_intervals[tgt_phys].append(
                        model.NewOptionalFixedSizeIntervalVar(peak_start, 1, is_selected, f"tgt_{g_id}_{k}")
                    )

                    if tgt_phys in completed_resource_usage:
                        peak_end = model.NewIntVar(0, horizon, f"peak_e_{g_id}_{k}")
                        model.Add(peak_end == peak_start + 1)
                        for (
                            comp_start,
                            comp_end,
                            comp_gate_id,
                        ) in completed_resource_usage[tgt_phys]:
                            b = model.NewBoolVar(f"no_overlap_tgt_{g_id}_{k}_{comp_gate_id}")
                            model.Add(peak_end <= comp_start).OnlyEnforceIf([is_selected, b])
                            model.Add(peak_start >= comp_end).OnlyEnforceIf([is_selected, b.Not()])

                    # Intermediate nodes (CNOT ladder body + post-ladder reset slot)
                    for h, node in enumerate(path):
                        if node in {src_phys, tgt_phys}:
                            continue
                        node_start_time = h - 1
                        node_dur = durations[k] - 2 * (h - 1) + 1

                        node_s = model.NewIntVar(0, horizon, f"int_s_{g_id}_{k}_{h}")
                        model.Add(node_s == start_var + node_start_time)
                        node_e = model.NewIntVar(0, horizon, f"int_e_{g_id}_{k}_{h}")
                        model.Add(node_e == node_s + node_dur)
                        node_intervals[node].append(
                            model.NewOptionalIntervalVar(
                                node_s,
                                node_dur,
                                node_e,
                                is_selected,
                                f"int_{g_id}_{k}_{node}",
                            )
                        )

                        if node in completed_resource_usage:
                            for (
                                comp_start,
                                comp_end,
                                comp_gate_id,
                            ) in completed_resource_usage[node]:
                                b = model.NewBoolVar(f"no_overlap_int_{g_id}_{k}_{h}_{comp_gate_id}")
                                model.Add(node_e <= comp_start).OnlyEnforceIf([is_selected, b])
                                model.Add(node_s >= comp_end).OnlyEnforceIf([is_selected, b.Not()])

        # Cascade failures for gates whose predecessors failed routing
        dep_fail_count = 0
        for gate in sorted_gate_window:
            g_id = gate["id"]
            if g_id not in gate_vars:
                continue
            for p in dep_graph.predecessors(g_id):
                if any(g["id"] == p for g in sorted_gate_window) and p not in gate_vars:
                    failed_gates.append({
                        "id": g_id,
                        "qubits": gate["qubits"],
                        "reason": f"Depends on failed gate {p}",
                    })
                    dep_fail_count += 1
                    gate_vars.pop(g_id, None)
                    break

        if failed_gates:
            routing_fails = len(failed_gates) - dep_fail_count
            self._log(f"  [ROUTING FAILED] {routing_fails} routing + {dep_fail_count} dep = {len(failed_gates)} total")
            return None

        # [DETERMINISM] Build NoOverlap constraints in deterministic node order
        for node in sorted(node_intervals.keys(), key=self._node_sort_key):
            interval_list = node_intervals[node]
            model.AddNoOverlap(interval_list)

        for gate in sorted_gate_window:
            g_id = gate["id"]
            if g_id not in gate_vars:
                continue
            for q in gate["qubits"]:
                if q in qubit_available_time:
                    model.Add(gate_vars[g_id]["start"] >= qubit_available_time[q])
            for p in dep_graph.predecessors(g_id):
                if p in gate_vars:
                    model.Add(gate_vars[g_id]["start"] >= gate_vars[p]["end"])
                elif p in completed_gates:
                    model.Add(gate_vars[g_id]["start"] >= completed_gates[p])

        makespan = model.NewIntVar(0, horizon, "makespan")
        if gate_vars:
            # [DETERMINISM] Iterate gate_vars in sorted key order
            model.AddMaxEquality(makespan, [gate_vars[k]["end"] for k in sorted(gate_vars.keys())])

        # ALAP lock scheduling: push lock starts as late as possible without increasing makespan
        # [DETERMINISM] Collect lock starts in sorted key order
        lock_gate_starts = [gate_vars[k]["start"] for k in sorted(gate_vars.keys()) if gate_vars[k].get("starts_lock")]
        if lock_gate_starts:
            makespan_weight = horizon * (len(gate_vars) + 1)
            lock_start_sum = model.NewIntVar(0, horizon * len(lock_gate_starts), "lock_sum")
            model.Add(lock_start_sum == sum(lock_gate_starts))
            model.Minimize(makespan * makespan_weight - lock_start_sum)
        else:
            model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.cp_solver_timeout
        # [DETERMINISM] Default: single worker for reproducibility
        solver.parameters.num_search_workers = self.cp_num_workers
        solver.parameters.random_seed = self.seed % (2**31)
        solver.parameters.cp_model_presolve = True
        status = solver.Solve(model)

        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            self._log(f"  [CP SOLVER FAILED] Status: {status}")
            return None

        results = {}
        for g_id, v in gate_vars.items():
            if v["type"] == "2q":
                path_k = solver.Value(v["path_idx"])
                results[g_id] = {
                    "start": solver.Value(v["start"]),
                    "end": solver.Value(v["end"]),
                    "path": v["paths"][path_k],
                    "type": "2q",
                    "source": v["source"],
                    "target": v["target"],
                }
            else:
                results[g_id] = {
                    "start": solver.Value(v["start"]),
                    "end": solver.Value(v["end"]),
                    "path": [v["u_phys"]],
                    "type": "1q",
                    "op": v["op"],
                    "clbits": v["clbits"],
                    "is_measure": v.get("is_measure", False),
                }
        return results

    def _execute_full_routing(self, circuit: QuantumCircuit, placement: dict) -> tuple[dict | None, int | None]:
        """Route the entire circuit using a sliding-window CP-SAT approach.

        1. Run macro ancilla scheduling to inject cycle-breaking edges.
        2. Feed gates through the window in topological order; after each
           solve, commit the earliest commit_size gates and refill.
        3. Track ancilla lock/unlock state (first-use->locked, measure->unlocked).

        Returns (schedule, makespan) or (None, None) on failure.
        """
        dep_graph, _, gate_list = self._build_dependency_graph(circuit)

        ancilla_info = self._analyze_ancilla_conflicts(gate_list, dep_graph, placement)
        if ancilla_info:
            macro_schedule = self._solve_macro_schedule(ancilla_info, dep_graph)
            if macro_schedule is None:
                self._log("  [MACRO FAILED] Placement has unresolvable ancilla conflicts")
                return None, None
            dep_graph = self._inject_cycle_breaking_edges(dep_graph, ancilla_info, macro_schedule)

        gate_lookup = {g["id"]: g for g in gate_list}
        topo_order = list(nx.topological_sort(dep_graph))
        queue = [gate_lookup[gid] for gid in topo_order if gid in gate_lookup]

        completed_gates = {}
        completed_resource_usage = defaultdict(list)
        qubit_available_time = {}
        final_schedule = []
        locked_ancillas = set()

        max_end_time = 0
        total_gates = len(queue)
        scheduled_count = 0

        # [DETERMINISM] Use an ordered list + membership set instead of a plain set.
        # Python sets have non-deterministic iteration order, which would cause
        # the gate ordering within each CP-SAT window to vary between runs.
        window_list: list[str] = []  # preserves insertion (topological) order
        window_member: set = set()  # O(1) membership check
        while len(window_list) < self.window_size and queue:
            g = queue.pop(0)
            gid = g["id"]
            if gid not in window_member:
                window_list.append(gid)
                window_member.add(gid)

        window_num = 0
        while window_list:
            window = [gate_lookup[gid] for gid in window_list]
            window_num += 1

            # --- Pre-window path analysis (helps diagnose routing failures) ---
            two_q_in_window = [g for g in window if g["num_qubits"] == 2]
            if two_q_in_window and window_num <= 5:  # log first 5 windows
                path_lens = []
                no_path_count = 0
                for g in two_q_in_window:
                    s, t = (
                        g.get("source", g["qubits"][0]),
                        g.get("target", g["qubits"][1]),
                    )
                    if s in placement and t in placement:
                        if nx.has_path(self.phys_graph, placement[s], placement[t]):
                            d = nx.shortest_path_length(self.phys_graph, placement[s], placement[t])
                            path_lens.append(d)
                        else:
                            no_path_count += 1
                if path_lens:
                    self._debug(
                        f"  [WINDOW {window_num}] {len(window)}g ({len(two_q_in_window)} 2Q) | "
                        f"raw path lengths: min={min(path_lens)} mean={np.mean(path_lens):.1f} "
                        f"max={max(path_lens)} | locked_anc={len(locked_ancillas)} | "
                        f"no_raw_path={no_path_count}"
                    )

            local_results = self._solve_cp_window(
                window,
                placement,
                dep_graph,
                completed_gates,
                completed_resource_usage,
                qubit_available_time,
                locked_ancillas,
            )

            if local_results is None:
                locked_info = f", locked_ancillas={len(locked_ancillas)}" if locked_ancillas else ""
                self._log(f"  [ROUTING FAILED] Window {window_num}={len(window)}{locked_info}")
                self._log(f"  Scheduled {scheduled_count}/{total_gates} gates before failure")
                # Detailed failure diagnostic: show which qubits are locked and where
                if locked_ancillas:
                    locked_locs = {a: placement.get(a, "?") for a in sorted(locked_ancillas)}
                    locked_degs = {
                        a: self.phys_graph.degree(placement[a]) for a in sorted(locked_ancillas) if a in placement
                    }
                    self._log(f"  [LOCKED DETAIL] {locked_locs}")
                    self._log(f"  [LOCKED DEGREES] {locked_degs}")
                return None, None

            if not local_results:
                self._log("  [WARNING] Empty results")
                break

            scheduled_by_start = sorted(
                [(gid, local_results[gid]["start"]) for gid in window_list if gid in local_results],
                key=operator.itemgetter(1),
            )
            to_commit_ids = (
                [gid for gid, _ in scheduled_by_start]
                if not queue
                else [gid for gid, _ in scheduled_by_start[: self.commit_size]]
            )

            for g_id in to_commit_ids:
                gate = gate_lookup[g_id]
                res = local_results[g_id]
                completed_gates[g_id] = res["end"]
                max_end_time = max(max_end_time, res["end"])

                path = res["path"]
                start_time, end_time = res["start"], res["end"]

                if res["type"] == "2q":
                    src_logical = res.get("source", gate.get("source", gate["qubits"][0]))
                    tgt_logical = res.get("target", gate.get("target", gate["qubits"][1]))
                    qubit_available_time[src_logical] = end_time
                    dist = len(path) - 1
                    qubit_available_time[tgt_logical] = start_time + dist

                    completed_resource_usage[path[0]].append((start_time, end_time, g_id))
                    peak_start = start_time + (dist - 1)
                    completed_resource_usage[path[-1]].append((peak_start, peak_start + 1, g_id))
                    # Intermediate nodes: full diamond interval + post-ladder reset slot
                    base_dur = 2 * (dist - 1) + 1
                    for h in range(1, dist):
                        node = path[h]
                        int_start = start_time + (h - 1)
                        int_dur = base_dur - 2 * (h - 1) + 1
                        completed_resource_usage[node].append((int_start, int_start + int_dur, g_id))
                else:
                    qubit_available_time[gate["qubits"][0]] = end_time
                    completed_resource_usage[path[0]].append((start_time, end_time, g_id))

                entry = {
                    "id": g_id,
                    "start": start_time,
                    "end": end_time,
                    "path": path,
                    "type": res["type"],
                    "logical_qubits": gate.get("qubits", []),
                    "ancillas": [q for q in gate.get("qubits", []) if q in self.ancilla_nodes],
                }
                if res["type"] == "1q":
                    entry.update({
                        "op": res["op"],
                        "clbits": res["clbits"],
                        "is_measure": res.get("is_measure", False),
                    })
                final_schedule.append(entry)
                scheduled_count += 1

            for g_id in to_commit_ids:
                gate = gate_lookup[g_id]
                res = local_results[g_id]
                locked_ancillas.update(self.gate_locks.get(g_id, []))
                if res["type"] == "1q":
                    op = res.get("op")
                    if op and hasattr(op, "name") and op.name == "measure":
                        locked_ancillas.difference_update(gate["qubits"])

            # [DETERMINISM] Remove committed gates while preserving order of remaining
            commit_set = set(to_commit_ids)
            window_list = [gid for gid in window_list if gid not in commit_set]
            window_member -= commit_set
            while len(window_list) < self.window_size and queue:
                g = queue.pop(0)
                gid = g["id"]
                if gid not in window_member:
                    window_list.append(gid)
                    window_member.add(gid)

        if scheduled_count < total_gates:
            self._log(f"  [INCOMPLETE] Only {scheduled_count}/{total_gates} gates scheduled")
            return None, None

        path_lens_2q = [len(e["path"]) - 1 for e in final_schedule if e["type"] == "2q" and len(e["path"]) > 1]
        ladder_durs = [2 * (pl - 1) + 1 for pl in path_lens_2q if pl > 0]
        n_1q = sum(1 for e in final_schedule if e["type"] == "1q")
        n_2q = len(path_lens_2q)
        if path_lens_2q:
            self._log(f"  [ROUTING DONE] {n_1q} 1Q + {n_2q} 2Q gates, makespan={max_end_time}, windows={window_num}")
            self._log(
                f"  [ROUTING DONE] 2Q path lengths: min={min(path_lens_2q)} "
                f"mean={np.mean(path_lens_2q):.1f} max={max(path_lens_2q)} | "
                f"ladder durations: min={min(ladder_durs)} mean={np.mean(ladder_durs):.1f} "
                f"max={max(ladder_durs)}"
            )

        return final_schedule, max_end_time

    def _analyze_contention(
        self, circuit: QuantumCircuit, placement: dict
    ) -> tuple[dict | None, int | None, dict[object, float]]:
        """Run routing and extract per-node contention penalties.

        Performs a full routing attempt and records, for each gate, how long
        it waited beyond its dependency-mandated earliest start (resource
        contention delay).  On routing failure, failed gates contribute a
        large penalty to the nodes on their candidate paths.

        IMPORTANT: _execute_full_routing is called first without any
        pre-processing to avoid side effects from duplicate
        _build_dependency_graph calls.

        Returns (schedule_or_None, makespan_or_None, bottleneck_penalties)
        where bottleneck_penalties is {physical_node: float}.
        """
        # Route first -- _execute_full_routing handles its own dep graph
        schedule, makespan = self._execute_full_routing(circuit, placement)

        # Now build dep graph and gate weights for contention analysis only
        dep_graph, _, gate_list = self._build_dependency_graph(circuit)
        self._compute_gate_weights(gate_list, dep_graph)
        gate_lookup = {g["id"]: g for g in gate_list}

        node_penalties = defaultdict(float)

        if schedule is not None:
            committed_end = {}
            qubit_avail = {}

            for entry in schedule:
                g_id = entry["id"]
                gate = gate_lookup.get(g_id)
                if gate is None:
                    continue

                actual_start = entry["start"]

                # Dependency-mandated earliest
                dep_earliest = 0
                for p in dep_graph.predecessors(g_id):
                    if p in committed_end:
                        dep_earliest = max(dep_earliest, committed_end[p])
                for q in gate.get("qubits", []):
                    if q in qubit_avail:
                        dep_earliest = max(dep_earliest, qubit_avail[q])

                contention_delay = max(0, actual_start - dep_earliest)

                if contention_delay > 0 and entry["type"] == "2q":
                    path = entry["path"]
                    weight = gate.get("weight", 1.0)
                    penalty = contention_delay * weight
                    for node in path:
                        node_penalties[node] += penalty

                committed_end[g_id] = entry["end"]
                if entry["type"] == "2q":
                    path = entry["path"]
                    dist = len(path) - 1
                    src_q = gate.get("source", gate["qubits"][0])
                    tgt_q = gate.get("target", gate["qubits"][1])
                    qubit_avail[src_q] = entry["end"]
                    qubit_avail[tgt_q] = entry["start"] + dist
                else:
                    qubit_avail[gate["qubits"][0]] = entry["end"]

        else:
            # Failed routing: use heavy penalties on qubits involved in
            # gates that couldn't be scheduled.  We identify high-contention
            # nodes by looking at which gates' physical positions are densely
            # packed in the placement.
            for gate in gate_list:
                if gate["num_qubits"] != 2:
                    continue
                src = gate.get("source", gate["qubits"][0])
                tgt = gate.get("target", gate["qubits"][1])
                if src not in placement or tgt not in placement:
                    continue
                ps, pt = placement[src], placement[tgt]
                # Add penalty to all nodes on shortest path
                if nx.has_path(self.phys_graph, ps, pt):
                    path = nx.shortest_path(self.phys_graph, ps, pt)
                    weight = gate.get("weight", 1.0)
                    penalty = weight * 10.0
                    for node in path[1:-1]:
                        node_penalties[node] += penalty

        return schedule, makespan, dict(node_penalties)

    # =========================================================================
    # 4. Output Circuit Generation
    # =========================================================================

    def generate_output_circuit(self, schedule_data: dict) -> tuple[QuantumCircuit, dict]:
        """Emit a Qiskit QuantumCircuit from the schedule.

        Translates each scheduled operation into CNOT-ladder sequences (2Q) or
        direct gate applications (1Q), inserting resets for:
          - Post-measurement ancilla cleanup (prevents contamination on reuse as route qubit).
          - Post-ladder intermediate cleanup: every intermediate (route) qubit is
            reset after the backward pass, using the slot reserved by the CP-SAT
            scheduler (+1 on each intermediate node interval).
        """
        phys_to_logical = {v: k for k, v in self.placement.items()}

        reg_groups = defaultdict(list)
        mapped_phys_nodes = set(phys_to_logical.keys())

        for p_node, log_name in phys_to_logical.items():
            last_us = log_name.rfind("_")
            if last_us != -1:
                reg_name = log_name[:last_us]
                try:
                    idx = int(log_name[last_us + 1 :])
                except ValueError:
                    idx = 0
            else:
                reg_name = "unknown"
                idx = 0
            reg_groups[reg_name].append((idx, p_node))

        route_nodes = [p_node for p_node in self.phys_nodes if p_node not in mapped_phys_nodes]

        regs = []
        node_to_qubit = {}

        for reg_name in sorted(reg_groups.keys()):
            group = sorted(reg_groups[reg_name], key=operator.itemgetter(0))
            reg_class = AncillaRegister if "anc" in reg_name.lower() else QuantumRegister
            q_reg = reg_class(len(group), reg_name)
            regs.append(q_reg)
            for i, (_, p_node) in enumerate(group):
                node_to_qubit[p_node] = q_reg[i]

        if route_nodes:
            route_nodes.sort(key=self._node_sort_key)
            r_reg = QuantumRegister(len(route_nodes), "route")
            regs.append(r_reg)
            for i, p_node in enumerate(route_nodes):
                node_to_qubit[p_node] = r_reg[i]

        cl_map = {}
        if self.input_clbits:
            c_reg = ClassicalRegister(len(self.input_clbits), "c")
            regs.append(c_reg)
            for i, old_c in enumerate(self.input_clbits):
                cl_map[old_c] = c_reg[i]

        qc = QuantumCircuit(*regs)
        sorted_ops = sorted(schedule_data, key=operator.itemgetter("start"))

        for op in sorted_ops:
            path = op["path"]
            qubits = [node_to_qubit[n] for n in path]

            if op["type"] == "2q":
                # Forward pass
                for i in range(len(qubits) - 2):
                    qc.cx(qubits[i], qubits[i + 1])

                # Peak CNOT
                if len(qubits) >= 2:
                    qc.cx(qubits[-2], qubits[-1])

                # Backward pass
                for i in range(len(qubits) - 3, -1, -1):
                    qc.cx(qubits[i], qubits[i + 1])

                # Scheduler has reserved the time slot for these resets
                for q in qubits[1:-1]:
                    qc.reset(q)

            elif op["type"] == "1q":
                inst = op["op"]
                qubit = qubits[0]

                if inst.name == "reset":
                    qc.reset(qubit)
                elif inst.name == "measure":
                    qc.append(inst, qubits, [cl_map[c] for c in op["clbits"]])
                    # Immediate post-measurement reset prevents contamination on ancilla reuse
                    qc.reset(qubit)
                else:
                    qc.append(inst, qubits)

        return qc, node_to_qubit

    @staticmethod
    def _generate_placement_strategies(num_rounds: int) -> list[tuple[str, float]]:
        """Produce a list of (layout_strategy, perturbation_scale) tuples for multi-round placement."""
        strategies = [("kamada_kawai", 0.0)]
        if num_rounds >= 2:
            strategies.append(("spectral", 0.0))
        remaining = num_rounds - len(strategies)
        for i in range(remaining):
            perturb = 0.1 + 0.5 * i / max(1, remaining - 1)
            strategies.append(("kamada_kawai", perturb))
        return strategies[:num_rounds]

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    @staticmethod
    def get_qubit_indices(circuit: QuantumCircuit) -> dict[str, set[int]]:
        """Classify output circuit qubits into route, ancilla, and data sets by register name."""
        result = {"route": set(), "ancilla": set(), "data": set()}
        for idx, qubit in enumerate(circuit.qubits):
            if circuit.find_bit(qubit).registers:
                reg_name = circuit.find_bit(qubit).registers[0][0].name.lower()
                if "route" in reg_name:
                    result["route"].add(idx)
                elif "anc" in reg_name:
                    result["ancilla"].add(idx)
                else:
                    result["data"].add(idx)
            else:
                result["data"].add(idx)
        return result

    def _route_with_crisp(self, circuit: QuantumCircuit, pmap: dict, seed_val: int) -> tuple[dict | None, int | None]:
        """Route a placement with optional CRISP contention-feedback refinement.

        Flow: route1 -> analyze contention -> swap2(from pi1, with penalties) -> route2
        Each iteration starts from the previous placement (not from scratch).
        Contention penalties accumulate across iterations (CRISP-style additive inflation).
        Best-seen tracking: keeps the best (schedule, makespan) across all attempts.

        Returns (best_schedule, best_makespan) -- same contract as _execute_full_routing.
        """
        # First routing attempt (no contention info)
        schedule_1, makespan_1, contention_1 = self._analyze_contention(circuit, pmap)

        if self.crisp_iterations <= 0:
            return schedule_1, makespan_1

        best_schedule = schedule_1
        best_makespan = makespan_1
        current_pmap = pmap

        # Accumulating bottleneck penalties (additive across iterations)
        accumulated_penalties = defaultdict(float)
        for node, penalty in contention_1.items():
            accumulated_penalties[node] += penalty

        logical_g = getattr(self, "_logical_g_weighted", None)
        if logical_g is None:
            # Fallback: rebuild (shouldn't happen in normal flow)
            dep_graph, pair_nodes, gate_list = self._build_dependency_graph(circuit)
            self._compute_gate_weights(gate_list, dep_graph)
            logical_g = self._create_weighted_graph(pair_nodes, gate_list)

        status_1 = "OK" if schedule_1 is not None else "FAIL"
        n_hot = sum(1 for v in accumulated_penalties.values() if v > 0)
        crisp_log_parts = [
            f"route1={status_1}{'(ms=' + str(best_makespan) + ')' if best_makespan else ''} contention_nodes={n_hot}"
        ]

        for ci in range(self.crisp_iterations):
            max_penalty = max(accumulated_penalties.values()) if accumulated_penalties else 1.0
            if max_penalty > 0:
                scale = self.layout_superlinear_coeff * self.crisp_penalty_scale / max_penalty
                normed = {n: v * scale for n, v in accumulated_penalties.items() if v > 0}
            else:
                normed = {}

            # Swap search from CURRENT placement with contention penalties
            crisp_seed = seed_val + 10000 * (ci + 1)
            refined_pmap, _ = self._refine_placement(
                current_pmap,
                logical_g,
                crisp_seed,
                bottleneck_penalties=normed,
                swap_budget=self.crisp_swap_budget,
                stall_limit=self.crisp_stall_limit,
            )

            schedule_r, makespan_r, contention_r = self._analyze_contention(circuit, refined_pmap)

            for node, penalty in contention_r.items():
                accumulated_penalties[node] += penalty

            status_r = "OK" if schedule_r is not None else "FAIL"
            n_hot = sum(1 for v in contention_r.values() if v > 0)

            improved = False
            if schedule_r is not None:
                if best_schedule is None:
                    # Recovered from failure!
                    improved = True
                elif makespan_r < best_makespan:
                    improved = True

            if improved:
                best_schedule = schedule_r
                best_makespan = makespan_r

            crisp_log_parts.append(
                f"crisp_{ci + 1}={status_r}"
                f"{'(ms=' + str(makespan_r) + ')' if makespan_r else ''}"
                f"{'*' if improved else ''} hot={n_hot}"
            )

            # Always continue from the latest placement (even if worse)
            # to explore further in the contention landscape
            current_pmap = refined_pmap

        self._log(f"  [CRISP] {' -> '.join(crisp_log_parts)}")
        return best_schedule, best_makespan

    def schedule_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Run the full scheduling pipeline: multi-round placement -> routing -> output generation.

        Selection criteria (lexicographic):
          1. post_selection: min layout cost, then min depth, then min op count.
          2. circuit_level: min makespan, then min depth, then min op count.

        If crisp_iterations > 0, each round includes contention-feedback
        refinement: route -> analyze contention -> swap with penalties -> re-route.

        Raises RuntimeError if no routable placement is found.
        """
        self._log(f"--- Initialization (Seed: {self.seed}) ---")
        if self.crisp_iterations > 0:
            self._log(f"    CRISP contention feedback: {self.crisp_iterations} iteration(s)")

        master_rng = np.random.RandomState(self.seed)
        round_seeds = master_rng.randint(0, 1_000_000, size=self.num_rounds)
        strategies = self._generate_placement_strategies(self.num_rounds)

        best_schedule = None

        if self.placement_strategy == "post_selection":
            self._log(f"--- Strategy: Post-Selection (Rounds: {self.num_rounds}) ---")
            valid_placements = []

            for i in range(self.num_rounds):
                seed_val = round_seeds[i]
                strategy, perturb = strategies[i]
                self._log(f"\n  Round {i + 1} ({strategy}, perturb={perturb:.2f}):")

                pmap, cost = self.run_placement_phase(circuit, seed_val, strategy, perturb)
                self._debug(f"    Placement cost: {cost:.2f}")

                test_schedule, makespan = self._route_with_crisp(circuit, pmap, seed_val)
                if test_schedule is None:
                    self._log(f"  Round {i + 1}: FAILED (not routable, cost={cost:.2f})")
                else:
                    self._log(f"  Round {i + 1}: SUCCESS (makespan={makespan}, cost={cost:.2f})")
                    valid_placements.append({
                        "placement": pmap,
                        "layout_cost": cost,
                        "schedule": test_schedule,
                        "makespan": makespan,
                    })

            if not valid_placements:
                msg = (
                    f"[CRITICAL] No routable placement found after {self.num_rounds} attempts! "
                    f"Try increasing num_rounds or check circuit/topology compatibility."
                )
                raise RuntimeError(msg)

            best_layout_cost = min(p["layout_cost"] for p in valid_placements)
            best_cost_placements = [p for p in valid_placements if p["layout_cost"] == best_layout_cost]

            if len(best_cost_placements) == 1:
                best = best_cost_placements[0]
                self.placement = best["placement"]
                best_schedule = best["schedule"]
                self._log(f"\n  > Selected: layout_cost={best['layout_cost']:.2f}, makespan={best['makespan']}")
            else:
                self._log(
                    f"\n  > {len(best_cost_placements)} tied at layout_cost={best_layout_cost:.2f}, evaluating circuits..."
                )
                candidates = []
                for p in best_cost_placements:
                    self.placement = p["placement"]
                    output_circuit, _ = self.generate_output_circuit(p["schedule"])
                    candidates.append({
                        "placement": p["placement"],
                        "schedule": p["schedule"],
                        "output_circuit": output_circuit,
                        "depth": output_circuit.depth(),
                        "op_count": len(output_circuit.data),
                        "makespan": p["makespan"],
                    })
                candidates.sort(key=operator.itemgetter("depth", "op_count"))
                best = candidates[0]
                self.placement = best["placement"]
                self._log(f"  > Selected: depth={best['depth']}, ops={best['op_count']}")
                return best["output_circuit"]

        elif self.placement_strategy == "circuit_level":
            self._log(f"--- Strategy: Circuit Level (Rounds: {self.num_rounds}) ---")
            valid_placements = []

            for i in range(self.num_rounds):
                seed_val = round_seeds[i]
                strategy, perturb = strategies[i]
                self._debug(f"  Round {i + 1} ({strategy}, perturb={perturb:.2f})...")

                pmap, cost = self.run_placement_phase(circuit, seed_val, strategy, perturb)
                schedule, makespan = self._route_with_crisp(circuit, pmap, seed_val)

                if schedule is None:
                    self._log(f"  Round {i + 1}: FAILED (not routable)")
                else:
                    self._log(f"  Round {i + 1}: SUCCESS (cost={cost:.2f}, makespan={makespan})")
                    valid_placements.append({
                        "placement": pmap,
                        "layout_cost": cost,
                        "schedule": schedule,
                        "makespan": makespan,
                    })

            if not valid_placements:
                msg = f"[CRITICAL] No routable placement found after {self.num_rounds} attempts!"
                raise RuntimeError(msg)

            best_makespan = min(p["makespan"] for p in valid_placements)
            best_makespan_placements = [p for p in valid_placements if p["makespan"] == best_makespan]

            self._log(
                f"\n  > {len(valid_placements)} routable, {len(best_makespan_placements)} at best makespan ({best_makespan})"
            )

            if len(best_makespan_placements) == 1:
                best = best_makespan_placements[0]
                self.placement = best["placement"]
                best_schedule = best["schedule"]
                output_circuit, _ = self.generate_output_circuit(best_schedule)
                self._log(f"  > Selected (depth={output_circuit.depth()}, ops={len(output_circuit.data)})")
                return output_circuit
            self._log("  > Evaluating output circuits for tie-breaking...")
            candidates = []
            for p in best_makespan_placements:
                self.placement = p["placement"]
                output_circuit, _ = self.generate_output_circuit(p["schedule"])
                candidates.append({
                    "placement": p["placement"],
                    "schedule": p["schedule"],
                    "output_circuit": output_circuit,
                    "depth": output_circuit.depth(),
                    "op_count": len(output_circuit.data),
                    "layout_cost": p["layout_cost"],
                })
                self._log(
                    f"    - depth={candidates[-1]['depth']}, ops={candidates[-1]['op_count']}, "
                    f"layout_cost={p['layout_cost']:.2f}"
                )
            candidates.sort(key=operator.itemgetter("depth", "op_count"))
            best = candidates[0]
            self.placement = best["placement"]
            self._log(f"  > Selected: depth={best['depth']}, ops={best['op_count']}")
            return best["output_circuit"]

        output_circuit, _ = self.generate_output_circuit(best_schedule)
        return output_circuit
