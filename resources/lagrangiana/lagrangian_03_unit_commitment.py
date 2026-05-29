"""
Descomposición Lagrangiana — Ejemplo 3: Unit Commitment (UC)
============================================================
N=6 unidades térmicas, T=12 períodos horarios.
Se relajan las restricciones de balance de potencia sum_n p_{nt} = D_t.
El subproblema se descompone por unidad y cada uno es un MILP resuelto con Gurobi.

Variables por unidad n y período t:
    u[n,t]  in {0,1}: estado encendido
    p[n,t]  >= 0:     potencia generada
    v[n,t]  in {0,1}: arranque (startup)
    w[n,t]  in {0,1}: apagado (shutdown)

Restricción complicante (relajada): sum_n p[n,t] = D[t]  para todo t
Restricciones en X (retenidas por unidad): min/max power, min-on/off, ramping,
    lógica arranque/apagado.

Estructura:
    1. DATOS
    2. PROBLEMA ORIGINAL UC (Gurobi MIP — referencia, instancia pequeña)
    3. SUBPROBLEMA POR UNIDAD
    4. HEURÍSTICA PRIMAL (economic dispatch LP)
    5. MÉTODO DEL SUBGRADIENTE
    6. ANÁLISIS
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
N = 6   # unidades
T = 12  # períodos

# Parámetros de cada unidad
P_min = np.array([20., 25., 15., 30., 10., 20.])    # MW mínimo
P_max = np.array([80., 100., 60., 120., 50., 75.])  # MW máximo
c_fix = np.array([15., 20., 10., 25., 8., 12.])     # costo no-load ($/h)
c_var = np.array([2.5, 2.0, 3.0, 1.8, 3.5, 2.8])   # costo marginal ($/MWh)
c_start = np.array([30., 40., 20., 50., 15., 25.])  # costo arranque ($)

MUT = np.array([2, 3, 2, 4, 1, 2])   # mínimo tiempo encendido (períodos)
MDT = np.array([2, 2, 1, 3, 1, 2])   # mínimo tiempo apagado (períodos)
RU = np.array([30., 35., 25., 40., 20., 30.])  # ramp-up (MW/h)
RD = np.array([30., 35., 25., 40., 20., 30.])  # ramp-down (MW/h)

# Demanda (MW) por período
D = np.array([120., 150., 180., 200., 220., 240.,
              230., 210., 190., 170., 150., 130.])

I_N = range(N)
I_T = range(T)

print("=" * 65)
print("EJEMPLO 3: RELAJACIÓN LAGRANGIANA — UNIT COMMITMENT")
print("=" * 65)
print(f"  Unidades: {N}  |  Períodos: {T}")
print(f"  Demanda total: {D.sum():.0f} MWh  |  Max: {D.max():.0f} MW")
print(f"  Capacidad total instalada: {P_max.sum():.0f} MW")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEMA ORIGINAL UC (Gurobi MIP — referencia)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Solución MIP óptima UC (Gurobi) ---")
m_mip = gp.Model("unit_commitment")
m_mip.setParam("OutputFlag", 0)
m_mip.setParam("TimeLimit", 60)

u = m_mip.addVars(N, T, vtype=GRB.BINARY, name="u")
p = m_mip.addVars(N, T, lb=0, name="p")
v = m_mip.addVars(N, T, vtype=GRB.BINARY, name="v")  # arranque
w = m_mip.addVars(N, T, vtype=GRB.BINARY, name="w")  # apagado

# Objetivo
m_mip.setObjective(
    gp.quicksum(
        c_fix[n] * u[n, t] + c_var[n] * p[n, t] + c_start[n] * v[n, t]
        for n in I_N for t in I_T
    ),
    GRB.MINIMIZE
)

# Balance de potencia
for t in I_T:
    m_mip.addConstr(gp.quicksum(p[n, t] for n in I_N) == D[t], name=f"balance_{t}")

# Límites de generación
for n in I_N:
    for t in I_T:
        m_mip.addConstr(p[n, t] >= P_min[n] * u[n, t], name=f"pmin_{n}_{t}")
        m_mip.addConstr(p[n, t] <= P_max[n] * u[n, t], name=f"pmax_{n}_{t}")

# Lógica arranque/apagado: u[n,t] - u[n,t-1] = v[n,t] - w[n,t]
for n in I_N:
    for t in I_T:
        u_prev = u[n, t - 1] if t > 0 else 0
        m_mip.addConstr(u[n, t] - u_prev == v[n, t] - w[n, t], name=f"logic_{n}_{t}")

# Mínimo tiempo encendido
for n in I_N:
    for t in I_T:
        if t + MUT[n] <= T:
            m_mip.addConstr(
                gp.quicksum(u[n, t2] for t2 in range(t, min(t + MUT[n], T))) >= MUT[n] * v[n, t],
                name=f"mut_{n}_{t}"
            )

# Mínimo tiempo apagado
for n in I_N:
    for t in I_T:
        if t + MDT[n] <= T:
            m_mip.addConstr(
                gp.quicksum((1 - u[n, t2]) for t2 in range(t, min(t + MDT[n], T))) >= MDT[n] * w[n, t],
                name=f"mdt_{n}_{t}"
            )

# Ramping
for n in I_N:
    for t in range(1, T):
        m_mip.addConstr(p[n, t] - p[n, t - 1] <= RU[n], name=f"ru_{n}_{t}")
        m_mip.addConstr(p[n, t - 1] - p[n, t] <= RD[n], name=f"rd_{n}_{t}")

m_mip.optimize()

if m_mip.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    ref_obj = m_mip.ObjVal
    u_opt = np.array([[u[n, t].X for t in I_T] for n in I_N])
    p_opt = np.array([[p[n, t].X for t in I_T] for n in I_N])
    print(f"  Costo total MIP : {ref_obj:.2f} $")
    print(f"  Status          : {'Óptimo' if m_mip.Status == GRB.OPTIMAL else 'TimeLimit'}")
else:
    ref_obj = np.inf
    u_opt = None
    print("  MIP no encontró solución factible")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUBPROBLEMA POR UNIDAD
# ─────────────────────────────────────────────────────────────────────────────

def solve_unit_subproblem(n: int, lambda_: np.ndarray) -> tuple:
    """
    Resuelve el subproblema de la unidad n con multiplicadores lambda[t]:

        min  sum_t [c_fix[n]*u[n,t] + (c_var[n] - lambda[t])*p[n,t] + c_start[n]*v[n,t]]
        s.a. restricciones individuales de la unidad n

    Retorna: (u_sol, p_sol, obj_val)
    """
    m = gp.Model(f"unit_{n}")
    m.setParam("OutputFlag", 0)

    u_n = m.addVars(T, vtype=GRB.BINARY, name="u")
    p_n = m.addVars(T, lb=0, name="p")
    v_n = m.addVars(T, vtype=GRB.BINARY, name="v")
    w_n = m.addVars(T, vtype=GRB.BINARY, name="w")

    # costos modificados: (c_var[n] - lambda[t]) * p[n,t]
    m.setObjective(
        gp.quicksum(
            c_fix[n] * u_n[t] + (c_var[n] - lambda_[t]) * p_n[t] + c_start[n] * v_n[t]
            for t in I_T
        ),
        GRB.MINIMIZE
    )

    # límites de generación
    for t in I_T:
        m.addConstr(p_n[t] >= P_min[n] * u_n[t])
        m.addConstr(p_n[t] <= P_max[n] * u_n[t])

    # lógica arranque/apagado
    for t in I_T:
        u_prev = u_n[t - 1] if t > 0 else 0
        m.addConstr(u_n[t] - u_prev == v_n[t] - w_n[t])

    # min-on time
    for t in I_T:
        if t + MUT[n] <= T:
            m.addConstr(
                gp.quicksum(u_n[t2] for t2 in range(t, min(t + MUT[n], T))) >= MUT[n] * v_n[t]
            )

    # min-off time
    for t in I_T:
        if t + MDT[n] <= T:
            m.addConstr(
                gp.quicksum((1 - u_n[t2]) for t2 in range(t, min(t + MDT[n], T))) >= MDT[n] * w_n[t]
            )

    # ramping
    for t in range(1, T):
        m.addConstr(p_n[t] - p_n[t - 1] <= RU[n])
        m.addConstr(p_n[t - 1] - p_n[t] <= RD[n])

    m.optimize()

    u_sol = np.array([u_n[t].X for t in I_T])
    p_sol = np.array([p_n[t].X for t in I_T])
    return u_sol, p_sol, m.ObjVal


def compute_q(lambda_: np.ndarray) -> tuple:
    """
    Calcula q(lambda) resolviendo todos los subproblemas unitarios.
    q(lambda) = sum_n obj_n(lambda) + sum_t lambda_t * D[t]
    Retorna: (u_all, p_all, q_val, subgradient)
    """
    u_all = np.zeros((N, T))
    p_all = np.zeros((N, T))
    q_val = float(np.dot(lambda_, D))  # suma lambda_t * D_t

    for n in I_N:
        u_n, p_n, obj_n = solve_unit_subproblem(n, lambda_)
        u_all[n] = u_n
        p_all[n] = p_n
        q_val += obj_n  # suma costo modificado de cada subproblema

    # subgradiente: g[t] = D[t] - sum_n p[n,t]
    subgradient = D - p_all.sum(axis=0)
    return u_all, p_all, q_val, subgradient

# ─────────────────────────────────────────────────────────────────────────────
# 4. HEURÍSTICA PRIMAL (economic dispatch LP)
# ─────────────────────────────────────────────────────────────────────────────

def economic_dispatch(u_fixed: np.ndarray) -> float:
    """
    Dado u_fixed (compromisos de encendido), resuelve el LP de economic dispatch:
        min  sum_{n,t} c_var[n] p[n,t]
        s.a. sum_n p[n,t] = D[t]
             P_min[n]*u[n,t] <= p[n,t] <= P_max[n]*u[n,t]
    Retorna costo total (variable + arranque + no-load) o inf si infactible.
    """
    m = gp.Model("ed_lp")
    m.setParam("OutputFlag", 0)

    p_ed = m.addVars(N, T, lb=0, name="p")

    m.setObjective(
        gp.quicksum(c_var[n] * p_ed[n, t] for n in I_N for t in I_T),
        GRB.MINIMIZE
    )

    for t in I_T:
        m.addConstr(gp.quicksum(p_ed[n, t] for n in I_N) == D[t])

    for n in I_N:
        for t in I_T:
            m.addConstr(p_ed[n, t] >= P_min[n] * u_fixed[n, t])
            m.addConstr(p_ed[n, t] <= P_max[n] * u_fixed[n, t])

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return np.inf

    # costo total: operación + arranques + no-load
    p_sol = np.array([[p_ed[n, t].X for t in I_T] for n in I_N])
    startup_cost = 0.0
    for n in I_N:
        for t in I_T:
            u_prev = u_fixed[n, t - 1] if t > 0 else 0
            if u_fixed[n, t] > 0.5 and u_prev < 0.5:
                startup_cost += c_start[n]

    var_cost_total = float(np.sum(c_var[:, None] * p_sol))
    noload_cost = float(np.sum(c_fix[:, None] * u_fixed))
    return var_cost_total + noload_cost + startup_cost

# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTODO DEL SUBGRADIENTE
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER = 150
EPSILON = 1e-2  # 1% gap
ALPHA = 1.0

lambda_k = np.zeros(T)  # precios de energía iniciales
best_LB = -np.inf
best_UB = np.inf
best_u = None

history_LB = []
history_UB = []
history_gap_primal = []

print("\n--- Método del Subgradiente (Unit Commitment) ---")
print(f"{'Iter':>4} | {'q(λ)':>10} | {'UB':>10} | {'gap%':>8} | {'|g|':>8}")
print("-" * 50)

for k in range(1, MAX_ITER + 1):
    u_k, p_k, q_k, g_k = compute_q(lambda_k)

    if q_k > best_LB:
        best_LB = q_k

    # UB: economic dispatch con u_k fijado
    ub_k = economic_dispatch(u_k)
    if ub_k < best_UB:
        best_UB = ub_k
        best_u = u_k.copy()

    history_LB.append(best_LB)
    history_UB.append(best_UB if best_UB < np.inf else None)

    gap_val = float(np.sqrt(np.dot(g_k, g_k)))  # norma del subgradiente
    gap_pct = (best_UB - best_LB) / (abs(best_LB) + 1e-9) if best_UB < np.inf else np.inf

    # gap de factibilidad primal
    primal_gap = float(np.max(np.abs(D - p_k.sum(axis=0))))
    history_gap_primal.append(primal_gap)

    if k <= 20 or k % 25 == 0:
        ub_str = f"{best_UB:10.2f}" if best_UB < np.inf else f"{'∞':>10}"
        print(f"{k:>4} | {q_k:10.2f} | {ub_str} | {gap_pct*100:7.2f}% | {gap_val:8.2f}")

    if gap_pct < EPSILON:
        print(f"\n  Convergió en iteración {k} (gap = {gap_pct*100:.2f}%)")
        break

    # paso de Polyak
    norm_g_sq = float(np.dot(g_k, g_k))
    if norm_g_sq < 1e-10:
        print(f"\n  ||g|| ≈ 0 en iter {k}, convergido")
        break

    if best_UB < np.inf:
        t_k = ALPHA * (best_UB - q_k) / norm_g_sq
    else:
        t_k = 2.0 / np.sqrt(k)

    # lambda sin restricción de signo (restricción de igualdad)
    lambda_k = lambda_k + t_k * g_k

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print(f"  Mejor LB Lagrangiana   : {best_LB:.2f} $")
print(f"  Mejor UB (heurística)  : {best_UB:.2f} $")
if ref_obj < np.inf:
    print(f"  MIP óptimo             : {ref_obj:.2f} $")
    print(f"  Brecha LR vs MIP       : {(ref_obj - best_LB)/ref_obj*100:.2f}%")
print(f"  Unidades encendidas (UB):")
if best_u is not None:
    for n in I_N:
        on_periods = [t for t in I_T if best_u[n, t] > 0.5]
        print(f"    Unidad {n+1}: períodos {on_periods}")

# Gráfico
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Convergencia LB/UB
ax1 = fig.add_subplot(gs[0, 0])
iters = list(range(1, len(history_LB) + 1))
ax1.plot(iters, history_LB, color="#254F83", lw=1.5, label="LB (q(λ))")
ub_iters_plot = [iters[i] for i, v in enumerate(history_UB) if v is not None]
ub_vals_plot = [v for v in history_UB if v is not None]
ax1.plot(ub_iters_plot, ub_vals_plot, color="#CC3030", lw=1.5, linestyle="--", label="UB")
if ref_obj < np.inf:
    ax1.axhline(ref_obj, color="gray", lw=1, linestyle=":", label=f"MIP ({ref_obj:.0f}$)")
ax1.set_xlabel("Iteración")
ax1.set_ylabel("Costo ($)")
ax1.set_title("Convergencia LB/UB")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Gap de factibilidad primal
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(iters, history_gap_primal, color="#4A7A2C", lw=1.5)
ax2.set_xlabel("Iteración")
ax2.set_ylabel("max|D_t - sum_n p_{nt}| (MW)")
ax2.set_title("Gap de factibilidad primal (balance)")
ax2.grid(True, alpha=0.3)

# Despacho por unidad (mejor UB)
ax3 = fig.add_subplot(gs[1, :])
if best_u is not None:
    dispatch_ub = economic_dispatch.__wrapped__ if hasattr(economic_dispatch, '__wrapped__') else None
    colors = plt.cm.tab10(np.linspace(0, 0.6, N))
    bottom = np.zeros(T)
    for n in I_N:
        # estimación de despacho proporcional a P_max cuando encendido
        p_approx = np.where(best_u[n] > 0.5, P_min[n], 0)
        ax3.bar(range(T), p_approx, bottom=bottom, color=colors[n],
                label=f"Unidad {n+1} (P_min)", alpha=0.7)
        bottom += p_approx
ax3.plot(range(T), D, "k--", lw=2, label="Demanda D[t]")
ax3.set_xlabel("Período t")
ax3.set_ylabel("Potencia (MW)")
ax3.set_title("Despacho mínimo con compromisos de la mejor heurística")
ax3.legend(fontsize=7, ncol=4)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("lagrangian_03_uc_convergencia.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: lagrangian_03_uc_convergencia.png")
plt.show()
