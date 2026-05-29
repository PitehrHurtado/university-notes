"""
Descomposición Lagrangiana — Ejemplo 2: Uncapacitated Facility Location (UFL)
==============================================================================
Se relajan las restricciones de demanda sum_j x_{ij} = 1.
El subproblema se descompone por instalación y cada uno se resuelve con Gurobi.
El método del subgradiente con paso de Polyak maximiza el dual Lagrangiano.

Variables:
    y_j  in {0,1}: abrir instalación j
    x_ij >= 0:     fracción de demanda del cliente i servida por instalación j

Restricciones complicantes (relajadas):  sum_j x_{ij} = 1  para todo i
Restricciones en X (retenidas):          x_{ij} <= y_j,  0 <= x_{ij},  y_j in {0,1}

Estructura:
    1. DATOS
    2. PROBLEMA ORIGINAL UFL (Gurobi MIP — referencia)
    3. SUBPROBLEMA POR INSTALACIÓN (con lambda)
    4. HEURÍSTICA PRIMAL (greedy asignación + LP)
    5. MÉTODO DEL SUBGRADIENTE
    6. RESULTADOS
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
n_clients = 8   # número de clientes
n_facilities = 5  # número de instalaciones

# Costos de transporte c[i,j]: cliente i a instalación j
transport_cost = rng.integers(2, 20, size=(n_clients, n_facilities)).astype(float)
# Costos fijos de apertura f[j]
fixed_cost = rng.integers(10, 40, size=n_facilities).astype(float)

print("=" * 65)
print("EJEMPLO 2: RELAJACIÓN LAGRANGIANA — FACILITY LOCATION (UFL)")
print("=" * 65)
print(f"  Clientes: {n_clients}  |  Instalaciones: {n_facilities}")
print(f"  Costos fijos f: {fixed_cost}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEMA ORIGINAL UFL (Gurobi MIP — referencia)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Solución MIP óptima (Gurobi) ---")
m_mip = gp.Model("ufl_mip")
m_mip.setParam("OutputFlag", 0)

I = range(n_clients)
J = range(n_facilities)

y_mip = m_mip.addVars(J, vtype=GRB.BINARY, name="y")
x_mip = m_mip.addVars(I, J, lb=0, ub=1, name="x")

m_mip.setObjective(
    gp.quicksum(fixed_cost[j] * y_mip[j] for j in J) +
    gp.quicksum(transport_cost[i, j] * x_mip[i, j] for i in I for j in J),
    GRB.MINIMIZE
)
# demanda satisfecha
for i in I:
    m_mip.addConstr(gp.quicksum(x_mip[i, j] for j in J) == 1, name=f"dem_{i}")
# asignación solo a instalaciones abiertas
for i in I:
    for j in J:
        m_mip.addConstr(x_mip[i, j] <= y_mip[j], name=f"link_{i}_{j}")

m_mip.optimize()
opt_obj = m_mip.ObjVal
y_opt = np.array([y_mip[j].X for j in J])
x_opt = np.array([[x_mip[i, j].X for j in J] for i in I])
print(f"  Valor óptimo MIP     : {opt_obj:.4f}")
print(f"  Instalaciones abiertas: {[j for j in J if y_opt[j] > 0.5]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUBPROBLEMA POR INSTALACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def solve_facility_subproblem(j: int, lambda_: np.ndarray) -> tuple:
    """
    Subproblema para instalación j con multiplicadores lambda[i]:

        min  f_j y_j + sum_i (c_{ij} - lambda_i) x_{ij}
        s.a.  x_{ij} <= y_j   para todo i
              x_{ij} in [0,1], y_j in {0,1}

    Se resuelve analíticamente:
        - Si y_j = 0: todos x_{ij} = 0, costo = 0
        - Si y_j = 1: x_{ij} = 1 si c_{ij} - lambda_i < 0, sino 0
                      costo = f_j + sum_i max(0, c_{ij} - lambda_i) ... esperar:
                      para cada i se toma x_{ij}=1 si (c_{ij}-lambda_i)<0
    Decidimos y_j = 1 si el costo neto de abrir < 0.
    Usamos Gurobi para no asumir nada sobre la estructura.
    """
    m = gp.Model(f"facility_{j}")
    m.setParam("OutputFlag", 0)

    y_j = m.addVar(vtype=GRB.BINARY, name="y")
    x_j = m.addVars(I, lb=0, ub=1, name="x")

    # costos modificados
    mod_cost = transport_cost[:, j] - lambda_  # c_{ij} - lambda_i
    m.setObjective(
        fixed_cost[j] * y_j + gp.quicksum(mod_cost[i] * x_j[i] for i in I),
        GRB.MINIMIZE
    )
    for i in I:
        m.addConstr(x_j[i] <= y_j, name=f"link_{i}")

    m.optimize()
    x_sol = np.array([x_j[i].X for i in I])
    y_sol = y_j.X
    return y_sol, x_sol, m.ObjVal


def solve_all_facilities(lambda_: np.ndarray) -> tuple:
    """
    Resuelve todos los subproblemas de instalaciones y calcula q(lambda).
    Retorna: (y_sub, x_sub, q_val, subgradient)
    """
    y_sub = np.zeros(n_facilities)
    x_sub = np.zeros((n_clients, n_facilities))
    q_val = -np.dot(lambda_, np.ones(n_clients))  # - lambda^T 1 (el término constante)
    # porque L = sum_j f_j y_j + sum_ij (c_ij - lambda_i) x_ij + sum_i lambda_i
    # q(lambda) = min_subprob + sum_i lambda_i

    for j in J:
        y_j, x_j, obj_j = solve_facility_subproblem(j, lambda_)
        y_sub[j] = y_j
        x_sub[:, j] = x_j
        q_val += obj_j  # suma los costos modificados de cada subproblema

    # corrección: q(lambda) = sum_j obj_j + sum_i lambda_i
    q_val += np.sum(lambda_)

    # subgradiente: g_i = 1 - sum_j x_{ij}
    subgradient = np.ones(n_clients) - x_sub.sum(axis=1)
    return y_sub, x_sub, q_val, subgradient

# ─────────────────────────────────────────────────────────────────────────────
# 4. HEURÍSTICA PRIMAL
# ─────────────────────────────────────────────────────────────────────────────

def greedy_ub(y_fixed: np.ndarray) -> float:
    """
    Dado un vector y de instalaciones (posiblemente fraccionario),
    abre las instalaciones con y_j > 0.5 y asigna cada cliente
    a la instalación abierta de menor costo de transporte.
    Retorna el costo total (o inf si ninguna instalación abierta).
    """
    open_j = [j for j in J if y_fixed[j] > 0.5]
    if not open_j:
        return np.inf

    total_cost = sum(fixed_cost[j] for j in open_j)
    for i in I:
        best = min(open_j, key=lambda j: transport_cost[i, j])
        total_cost += transport_cost[i, best]
    return float(total_cost)


def lp_ub(y_fixed: np.ndarray) -> float:
    """
    Con instalaciones abiertas fijadas, resuelve LP de asignación con Gurobi.
    Esto da la mejor UB dado y_fixed.
    """
    open_j = [j for j in J if y_fixed[j] > 0.5]
    if not open_j:
        return np.inf

    m = gp.Model("lp_assign")
    m.setParam("OutputFlag", 0)
    x = m.addVars(I, open_j, lb=0, ub=1, name="x")
    m.setObjective(
        gp.quicksum(transport_cost[i, j] * x[i, j] for i in I for j in open_j),
        GRB.MINIMIZE
    )
    for i in I:
        m.addConstr(gp.quicksum(x[i, j] for j in open_j) == 1, name=f"dem_{i}")
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return np.inf
    transport_total = m.ObjVal
    fixed_total = sum(fixed_cost[j] for j in open_j)
    return transport_total + fixed_total

# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTODO DEL SUBGRADIENTE
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER = 300
EPSILON = 1e-3
ALPHA = 1.0  # factor paso de Polyak

lambda_k = np.zeros(n_clients)  # multiplicadores iniciales
best_LB = -np.inf
best_UB = np.inf
best_y = None

history_LB = []
history_UB = []

print("\n--- Método del Subgradiente ---")
print(f"{'Iter':>4} | {'q(λ)':>10} | {'UB':>10} | {'gap%':>8}")
print("-" * 42)

for k in range(1, MAX_ITER + 1):
    y_k, x_k, q_k, g_k = solve_all_facilities(lambda_k)

    if q_k > best_LB:
        best_LB = q_k

    # UB greedy
    ub_greedy = greedy_ub(y_k)
    if ub_greedy < best_UB:
        best_UB = ub_greedy
        best_y = y_k.copy()

    history_LB.append(best_LB)
    history_UB.append(best_UB if best_UB < np.inf else None)

    gap = (best_UB - best_LB) / (abs(best_LB) + 1e-9) if best_UB < np.inf else np.inf

    if k <= 15 or k % 30 == 0:
        ub_str = f"{best_UB:10.4f}" if best_UB < np.inf else f"{'∞':>10}"
        print(f"{k:>4} | {q_k:10.4f} | {ub_str} | {gap*100:7.2f}%")

    if gap < EPSILON:
        print(f"\n  Convergió en iteración {k}")
        break

    # paso de Polyak
    norm_g_sq = float(np.dot(g_k, g_k))
    if norm_g_sq < 1e-12:
        print(f"\n  Subgradiente cero en iter {k} — convergido")
        break

    if best_UB < np.inf:
        t_k = ALPHA * (best_UB - q_k) / norm_g_sq
    else:
        t_k = 1.0 / np.sqrt(k)

    # lambda sin restricción de signo (restricción de igualdad => lambda libre)
    lambda_k = lambda_k + t_k * g_k

# ─────────────────────────────────────────────────────────────────────────────
# 6. RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

# Heurística LP con la mejor y encontrada
ub_lp = lp_ub(best_y) if best_y is not None else np.inf

print("\n" + "=" * 65)
print("RESUMEN DE RESULTADOS")
print("=" * 65)
print(f"  Mejor LB Lagrangiana       : {best_LB:.4f}")
print(f"  Mejor UB (greedy)          : {best_UB:.4f}")
print(f"  Mejor UB (LP asignación)   : {ub_lp:.4f}")
print(f"  MIP óptimo                 : {opt_obj:.4f}")
print(f"  Instalaciones abiertas LR  : {[j for j in J if best_y is not None and best_y[j] > 0.5]}")
print(f"  Gap LR vs MIP              : {(opt_obj - best_LB)/opt_obj*100:.2f}%")

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
iters = list(range(1, len(history_LB) + 1))
axes[0].plot(iters, history_LB, color="#254F83", lw=1.5, label="LB (q(λ))")
ub_iters_plot = [iters[i] for i, v in enumerate(history_UB) if v is not None]
ub_vals_plot = [v for v in history_UB if v is not None]
axes[0].plot(ub_iters_plot, ub_vals_plot, color="#CC3030", lw=1.5, linestyle="--", label="UB (greedy)")
axes[0].axhline(opt_obj, color="gray", lw=1, linestyle=":", label=f"MIP óptimo ({opt_obj:.1f})")
axes[0].set_xlabel("Iteración")
axes[0].set_ylabel("Valor")
axes[0].set_title("Convergencia — UFL Lagrangiana")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Heatmap costos de transporte
im = axes[1].imshow(transport_cost, cmap="YlOrRd", aspect="auto")
axes[1].set_xlabel("Instalación j")
axes[1].set_ylabel("Cliente i")
axes[1].set_title("Costos de transporte c[i,j]")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("lagrangian_02_facility_convergencia.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: lagrangian_02_facility_convergencia.png")
plt.show()
