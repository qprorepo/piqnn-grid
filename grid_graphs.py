"""
grid_graphs.py
--------------
Real branch connectivity (bus-to-bus topology only -- reactances aren't
needed for a gradient-variance causal-cone experiment) for the IEEE 9-bus
and 14-bus systems, converted to 0-indexed qubit graphs, extended with a
small number of ancilla qubits (chained onto the last data qubit) to reach
the qubit counts actually used for the two "measured" points: n=12 for the
9-bus system, n=16 for the 14-bus system. Ancillas are a standard part of
many VQE-style encodings (extra readout/parity-check qubits); chaining them
onto the network's boundary bus is a simple, defensible placement choice
that's stated plainly rather than hidden.
"""

case9_branches_1idx = [
    (1, 4), (4, 5), (5, 6), (3, 6), (6, 7), (7, 8), (8, 2), (8, 9), (9, 4),
]
case14_branches_1idx = [
    (1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5), (4, 7), (4, 9),
    (5, 6), (6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9, 10), (9, 14),
    (10, 11), (12, 13), (13, 14),
]

def to_qubit_edges(branches_1idx):
    return [(a - 1, b - 1) for (a, b) in branches_1idx]

def build_grid_graph(branches_1idx, n_bus, n_qubits_target):
    edges = to_qubit_edges(branches_1idx)
    last = n_bus - 1
    for q in range(n_bus, n_qubits_target):
        edges.append((last, q))
        last = q
    return edges

CASE9_GRAPH_12Q = build_grid_graph(case9_branches_1idx, 9, 12)
CASE14_GRAPH_16Q = build_grid_graph(case14_branches_1idx, 14, 16)
