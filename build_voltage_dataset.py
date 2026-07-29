import csv
import numpy as np
from collections import deque
from swing_dynamics import MachineSystem


def bus_graph_hops_to_nearest_gen(case, idx, gen_buses):
    n = case["bus"].shape[0]
    adj = {k: [] for k in range(n)}
    for row in case["branch"]:
        f, t = idx[int(row[0])], idx[int(row[1])]
        adj[f].append(t)
        adj[t].append(f)
    gen_idx = set(idx[b] for b in gen_buses)
    dist = {k: None for k in range(n)}
    dq = deque()
    for g in gen_idx:
        dist[g] = 0
        dq.append(g)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist  # bus-index (0-based) -> hop distance to nearest generator


if __name__ == "__main__":
    sys9 = MachineSystem("case9.txt", H_override=np.array([23.64, 6.40, 3.01]))
    hops = bus_graph_hops_to_nearest_gen(sys9.case, sys9.idx, sys9.gen_buses)
    n_bus = sys9.case["bus"].shape[0]
    Pd = sys9.case["bus"][:, 2] / sys9.case["baseMVA"]
    Qd = sys9.case["bus"][:, 3] / sys9.case["baseMVA"]
    branch_x = {(int(r[0]), int(r[1])): r[3] for r in sys9.case["branch"]}

    rows = []
    for row in sys9.case["branch"]:
        fbus, tbus, x = int(row[0]), int(row[1]), row[3]
        for fault_at in (fbus, tbus):
            Yf_full = sys9.fault_full_ybus(fault_at)
            V_N = sys9.solve_bus_voltages(Yf_full, sys9.delta0)
            vmin = float(np.min(np.abs(V_N)))
            k = sys9.idx[fault_at]
            rows.append(dict(
                fault_bus=fault_at,
                bus_norm=(fault_at - 1) / (n_bus - 1) * 2 * np.pi - np.pi,
                branch_x=x,
                local_pd=Pd[k], local_qd=Qd[k],
                hops_to_gen=hops[k],
                vmin=vmin,
            ))

    vs = [r["vmin"] for r in rows]
    print(f"n_samples={len(rows)}  vmin range=[{min(vs):.3f}, {max(vs):.3f}]  mean={np.mean(vs):.3f}")

    with open("voltage_dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote voltage_dataset.csv")
