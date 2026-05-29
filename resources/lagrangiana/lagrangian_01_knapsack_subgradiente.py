"""
Descomposición Lagrangiana — Ejemplo 1: Multi-dimensional Knapsack
==================================================================
Se relaja una de las dos restricciones de capacidad del knapsack.
El subproblema resultante (knapsack 1D) se resuelve con Gurobi.
El método del subgradiente con paso de Polyak maximiza el dual Lagrangiano.

Estructura:
    1. DATOS
    2. PROBLEMA ORIGINAL (referencia con Gurobi)
    3. SUBPROBLEMA LAGRANGIANO
    4. MÉTODO DEL SUBGRADIENTE
    5. ANÁLISIS Y COMPARACIÓN DE COTAS
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 10  # número de ítems

profits = np.array([6, 9, 4, 7, 5, 8, 3, 10, 2, 6], dtype=float)  # c
weights_1 = np.array([4, 5, 3, 6, 2, 7, 1, 8, 3, 4], dtype=float)  # primera mochila (X)
weights_2 = np.array([3, 4, 2, 5, 3, 4, 2, 6, 1, 3], dtype=float)  # segunda mochila (relajada)
capacity_1 = 20.0  # capacidad mochila 1 (mantenida en X)
capacity_2 = 18.0  # capacidad mochila 2 (restricción complicante)

print("=" * 60)
print("EJEMPLO 1: RELAJACIÓN LAGRANGIANA — KNAPSACK 2D")
print("=" * 60)
print(f"Ítems: {n}  |  Cap1 (retenida): {capacity_1}  |  Cap2 (relajada): {capacity_2}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEMA ORIGINAL (referencia con Gurobi)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Solución MIP óptima (referencia Gurobi) ---")
m_ref = gp.Model("knapsack_2d")
m_ref.setParam("OutputFlag", 0)

x_ref = m_ref.addVars(n, vtype=GRB.BINARY, name="x")
m_ref.setObjective(gp.quicksum(profits[i] * x_ref[i] for i in range(n)), GRB.MAXIMIZE)
m_ref.addConstr(gp.quicksum(weights_1[i] * x_ref[i] for i in range(n)) <= capacity_1, "cap1")
m_ref.addConstr(gp.quicksum(weights_2[i] * x_ref[i] for i in range(n)) <= capacity_2, "cap2")
m_ref.optimize()

opt_obj = m_ref.ObjVal
x_opt = np.array([x_ref[i].X for i in range(n)])
print(f"  Valor óptimo MIP : {opt_obj:.4f}")
print(f"  Ítems seleccionados: {[i for i in range(n) if x_opt[i] > 0.5]}")

# LP-relaxation para comparación
m_lp = gp.Model("knapsack_lp")
m_lp.setParam("OutputFlag", 0)
x_lp = m_lp.addVars(n, lb=0, ub=1, name="x")
m_lp.setObjective(gp.quicksum(profits[i] * x_lp[i] for i in range(n)), GRB.MAXIMIZE)
m_lp.addConstr(gp.quicksum(weights_1[i] * x_lp[i] for i in range(n)) <= capacity_1, "cap1")
m_lp.addConstr(gp.quicksum(weights_2[i] * x_lp[i] for i in range(n)) <= capacity_2, "cap2")
m_lp.optimize()
lp_bound = m_lp.ObjVal
print(f"  LP-relajación      : {lp_bound:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUBPROBLEMA LAGRANGIANO
# ─────────────────────────────────────────────────────────────────────────────

def solve_subproblem(lambda_: float) -> tuple:
    """
    Resuelve LR(lambda): max sum_i (profits_i - lambda * weights_2_i) x_i
                          s.a. sum_i weights_1_i x_i <= cap_1, x_i in {0,1}
    Retorna: (x_sol, obj_lr, subgradient)
        subgradient = sum_i weights_2_i x_i - cap_2
    """
    # costos modificados: beneficio neto con el multiplicador
    modified_profits = profits - lambda_ * weights_2

    m = gp.Model("subproblem")
    m.setParam("OutputFlag", 0)

    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(modified_profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(weights_1[i] * x[i] for i in range(n)) <= capacity_1, "cap1")
    m.optimize()

    x_sol = np.array([x[i].X for i in range(n)])
    # valor dual: max f(x,lambda) = obj_subproblema - lambda * cap_2
    obj_lr = m.ObjVal - lambda_ * capacity_2
    # subgradiente: g = sum_i w2_i x_i - cap_2
    subgradient = float(np.dot(weights_2, x_sol) - capacity_2)
    return x_sol, obj_lr, subgradient


def compute_ub(x_candidate: np.ndarray) -> float:
    """
    Cota superior: si x_candidate es infactible para cap2, aplica reparación greedy.
    Devuelve el beneficio de la mejor solución factible encontrada.
    """
    x_round = x_candidate.copy()
    # reparar cap2: eliminar ítems con peor ratio beneficio/peso_2 hasta factible
    while np.dot(weights_2, x_round) > capacity_2 + 1e-6:
        active = np.where(x_round > 0.5)[0]
        if len(active) == 0:
            break
        # eliminar ítem con peor ratio beneficio/peso_2
        ratios = profits[active] / (weights_2[active] + 1e-9)
        worst = active[np.argmin(ratios)]
        x_round[worst] = 0
    # verificar factibilidad cap1 también
    if np.dot(weights_1, x_round) > capacity_1 + 1e-6:
        return -np.inf
    return float(np.dot(profits, x_round))

# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTODO DEL SUBGRADIENTE
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER = 200
EPSILON = 1e-4  # tolerancia de convergencia relativa
ALPHA = 1.0     # factor de paso de Polyak (en (0, 2])

lambda_k = 0.0  # multiplicador inicial (lambda >= 0 porque cap2 es <=)
best_LB = -np.inf
best_UB = np.inf
best_lambda = 0.0

history_LB = []
history_UB = []
history_lambda = []

print("\n--- Método del Subgradiente ---")
print(f"{'Iter':>4} | {'lambda':>8} | {'q(lambda)':>10} | {'UB':>10} | {'gap%':>8}")
print("-" * 50)

for k in range(1, MAX_ITER + 1):
    x_k, q_k, g_k = solve_subproblem(lambda_k)

    # actualizar cota inferior
    if q_k > best_LB:
        best_LB = q_k
        best_lambda = lambda_k

    # cota superior desde heurística primal
    ub_k = compute_ub(x_k)
    if ub_k > -np.inf and ub_k < best_UB:
        best_UB = ub_k

    history_LB.append(best_LB)
    history_UB.append(best_UB if best_UB < np.inf else None)
    history_lambda.append(lambda_k)

    gap = (best_UB - best_LB) / (abs(best_LB) + 1e-9) if best_UB < np.inf else np.inf

    if k <= 20 or k % 20 == 0:
        ub_str = f"{best_UB:.4f}" if best_UB < np.inf else "   ∞"
        print(f"{k:>4} | {lambda_k:>8.4f} | {q_k:>10.4f} | {ub_str:>10} | {gap*100:>7.2f}%")

    if gap < EPSILON:
        print(f"\n  Convergió en iteración {k} (gap = {gap*100:.4f}%)")
        break

    # paso de Polyak: t = alpha * (UB - q) / ||g||^2
    if abs(g_k) < 1e-10:
        break
    if best_UB < np.inf:
        t_k = ALPHA * (best_UB - q_k) / (g_k ** 2)
    else:
        t_k = ALPHA / (k ** 0.5)  # paso decreciente si no hay UB

    # lambda >= 0 (la restricción es de tipo <=)
    lambda_k = max(0.0, lambda_k + t_k * g_k)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS Y COMPARACIÓN DE COTAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN DE COTAS")
print("=" * 60)
print(f"  LP-relajación         : {lp_bound:.4f}")
print(f"  Mejor LB Lagrangiana  : {best_LB:.4f}")
print(f"  Mejor UB (heurística) : {best_UB:.4f}")
print(f"  MIP óptimo            : {opt_obj:.4f}")
print(f"  Brecha LR vs MIP      : {(opt_obj - best_LB):.4f} ({(opt_obj - best_LB)/opt_obj*100:.2f}%)")

# Gráfico de convergencia
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

iters = list(range(1, len(history_LB) + 1))
axes[0].plot(iters, history_LB, color="#254F83", lw=1.5, label="LB (q(λ))")
ub_vals = [v for v in history_UB if v is not None]
ub_iters = [iters[i] for i, v in enumerate(history_UB) if v is not None]
axes[0].plot(ub_iters, ub_vals, color="#CC3030", lw=1.5, linestyle="--", label="UB (heurística)")
axes[0].axhline(opt_obj, color="gray", lw=1, linestyle=":", label=f"MIP óptimo ({opt_obj:.1f})")
axes[0].set_xlabel("Iteración")
axes[0].set_ylabel("Valor de la cota")
axes[0].set_title("Convergencia del método del subgradiente")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, history_lambda, color="#4A7A2C", lw=1.5)
axes[1].set_xlabel("Iteración")
axes[1].set_ylabel("λ (multiplicador)")
axes[1].set_title("Evolución del multiplicador Lagrangiano")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lagrangian_01_convergencia.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: lagrangian_01_convergencia.png")
plt.show()
