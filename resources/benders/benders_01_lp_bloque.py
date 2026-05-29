"""
Descomposición de Benders — Ejemplo 1: LP con estructura de bloque
==================================================================
Problema LP 4×4 pedagógico con 2 restricciones de acoplamiento.
Las variables "complicantes" y son continuas aquí para la exposición.
El algoritmo de Benders se implementa de forma completamente manual.

Estructura:
    1. DATOS
    2. PROBLEMA ORIGINAL (referencia con Gurobi)
    3. FUNCIONES AUXILIARES (Master y Subproblema)
    4. LOOP PRINCIPAL DE BENDERS
    5. ANÁLISIS Y VERIFICACIÓN
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
# Problema: min c1^T x + c2^T y
#   s.a.  A1 x + A2 y >= b  (2 restricciones de acoplamiento)
#         x >= 0            (variables de recourse, 4-dim)
#         y >= 0, y <= 5    (variables complicantes, 2-dim)

# Costos de recourse
c1 = np.array([1.0, 2.0, 1.5, 1.0])
# Costos de variables complicantes
c2 = np.array([3.0, 2.0])

# Matriz del subproblema (A1 x >= b - A2 y)
A1 = np.array([
    [2.0, 1.0, 0.5, 1.0],
    [1.0, 2.0, 1.0, 0.5],
])  # 2 x 4
A2 = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
])  # 2 x 2
b = np.array([8.0, 7.0])

# Dominio de y
y_lb = np.zeros(2)
y_ub = np.array([5.0, 5.0])

print("=" * 60)
print("EJEMPLO BENDERS 1: LP CON ESTRUCTURA DE BLOQUE")
print("=" * 60)
print(f"  Variables de recourse x: {len(c1)}-dim")
print(f"  Variables complicantes y: {len(c2)}-dim")
print(f"  Restricciones de acoplamiento: {len(b)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEMA ORIGINAL (referencia con Gurobi)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Solución directa con Gurobi (referencia) ---")
m_full = gp.Model("lp_full")
m_full.setParam("OutputFlag", 0)

n_x, n_y = len(c1), len(c2)
x_full = m_full.addVars(n_x, lb=0, name="x")
y_full = m_full.addVars(n_y, lb=0, ub=5, name="y")

m_full.setObjective(
    gp.quicksum(c1[i] * x_full[i] for i in range(n_x)) +
    gp.quicksum(c2[j] * y_full[j] for j in range(n_y)),
    GRB.MINIMIZE
)

for row in range(len(b)):
    m_full.addConstr(
        gp.quicksum(A1[row, i] * x_full[i] for i in range(n_x)) +
        gp.quicksum(A2[row, j] * y_full[j] for j in range(n_y)) >= b[row],
        name=f"coupling_{row}"
    )

m_full.optimize()
ref_obj = m_full.ObjVal
ref_y = np.array([y_full[j].X for j in range(n_y)])
ref_x = np.array([x_full[i].X for i in range(n_x)])
print(f"  Valor óptimo directo : {ref_obj:.6f}")
print(f"  y*  = {ref_y}")
print(f"  x*  = {ref_x}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def solve_master(cuts_opt: list, cuts_feas: list) -> tuple:
    """
    Resuelve el Restricted Master Problem (RMP):
        min  c2^T y + eta
        s.a. [cortes de optimalidad]
             [cortes de factibilidad]
             y_lb <= y <= y_ub
             eta >= -1e6  (sin cota inferior fuerte al inicio)

    Retorna: (y_val, eta_val, obj_val)
    """
    m = gp.Model("rmp")
    m.setParam("OutputFlag", 0)

    y = m.addVars(n_y, lb=y_lb, ub=y_ub, name="y")
    eta = m.addVar(lb=-1e6, name="eta")

    m.setObjective(
        gp.quicksum(c2[j] * y[j] for j in range(n_y)) + eta,
        GRB.MINIMIZE
    )

    # cortes de optimalidad: eta >= u^T (b - A2 y)
    for u, rhs_const in cuts_opt:
        # u^T (b - A2 y) = u^T b - u^T A2 y
        m.addConstr(
            eta >= sum(u[i] * b[i] for i in range(len(b))) -
                   sum(u[i] * gp.quicksum(A2[i, j] * y[j] for j in range(n_y))
                       for i in range(len(b))),
            name=f"opt_cut_{len(cuts_opt)}"
        )

    # cortes de factibilidad: r^T (b - A2 y) <= 0
    for r, in cuts_feas:
        m.addConstr(
            sum(r[i] * b[i] for i in range(len(b))) -
            sum(r[i] * gp.quicksum(A2[i, j] * y[j] for j in range(n_y))
                for i in range(len(b))) <= 0,
            name=f"feas_cut_{len(cuts_feas)}"
        )

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"RMP no óptimo (Status={m.Status})")

    y_val = np.array([y[j].X for j in range(n_y)])
    return y_val, eta.X, m.ObjVal


def solve_subproblem(y_fixed: np.ndarray) -> dict:
    """
    Resuelve SP(y_fixed): min c1^T x  s.a.  A1 x >= b - A2 y_fixed, x >= 0
    y su dual DSP(y_fixed):  max u^T (b - A2 y*)  s.a.  A1^T u <= c1, u >= 0

    Retorna dict con claves:
        'status': 'optimal' o 'infeasible'
        'obj': valor óptimo de SP (si factible)
        'u':   vector dual (si factible), shadow prices de las restricciones
        'farkas': rayo de Farkas (si infactible)
    """
    rhs = b - A2 @ y_fixed  # b - A2 y*

    m = gp.Model("subproblem")
    m.setParam("OutputFlag", 0)
    m.setParam("InfUnbdInfo", 1)  # necesario para obtener FarkasDual

    x = m.addVars(n_x, lb=0, name="x")
    constrs = []
    for row in range(len(b)):
        c = m.addConstr(
            gp.quicksum(A1[row, i] * x[i] for i in range(n_x)) >= rhs[row],
            name=f"row_{row}"
        )
        constrs.append(c)

    m.setObjective(gp.quicksum(c1[i] * x[i] for i in range(n_x)), GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        u = np.array([c.Pi for c in constrs])
        return {'status': 'optimal', 'obj': m.ObjVal, 'u': u, 'farkas': None}
    elif m.Status == GRB.INFEASIBLE:
        # FarkasDual: certificado de infactibilidad (rayo del dual)
        farkas = np.array([c.FarkasDual for c in constrs])
        return {'status': 'infeasible', 'obj': None, 'u': None, 'farkas': farkas}
    else:
        raise RuntimeError(f"SP status inesperado: {m.Status}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOOP PRINCIPAL DE BENDERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER = 50
EPSILON = 1e-5

cuts_opt = []   # lista de (u, rhs_const)
cuts_feas = []  # lista de (r,)

LB = -np.inf
UB = np.inf
history_LB = []
history_UB = []

print("\n--- Algoritmo de Benders ---")
print(f"{'Iter':>4} | {'LB':>12} | {'UB':>12} | {'gap':>10} | {'corte'}")
print("-" * 60)

benders_sol_y = None

for k in range(1, MAX_ITER + 1):
    # Paso 1: resolver RMP
    y_k, eta_k, rmp_obj = solve_master(cuts_opt, cuts_feas)
    LB = rmp_obj

    # Paso 2: resolver subproblema SP(y^k) y su dual
    sp = solve_subproblem(y_k)

    if sp['status'] == 'optimal':
        u_k = sp['u']
        # UB real: costo total con y^k y x óptimo de SP(y^k)
        ub_candidate = float(c2 @ y_k) + sp['obj']
        if ub_candidate < UB:
            UB = ub_candidate
            benders_sol_y = y_k.copy()

        cut_type = "optimalidad"
        # corte: eta >= u^T (b - A2 y)
        cuts_opt.append((u_k, None))  # rhs se recalcula en solve_master usando b, A2

    else:  # infactible
        r_k = sp['farkas']
        cut_type = "factibilidad"
        cuts_feas.append((r_k,))

    history_LB.append(LB)
    history_UB.append(UB if UB < np.inf else None)

    gap = (UB - LB) / (abs(LB) + 1e-9) if UB < np.inf else np.inf
    ub_str = f"{UB:12.6f}" if UB < np.inf else f"{'∞':>12}"
    print(f"{k:>4} | {LB:12.6f} | {ub_str} | {gap*100:9.4f}% | {cut_type}")

    if gap < EPSILON:
        print(f"\n  Convergió en iteración {k} (gap = {gap:.2e})")
        break

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS Y VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICACIÓN")
print("=" * 60)
print(f"  Solución Benders y*  : {benders_sol_y}")
print(f"  Objetivo Benders     : {UB:.6f}")
print(f"  Objetivo Gurobi dir. : {ref_obj:.6f}")
print(f"  Diferencia           : {abs(UB - ref_obj):.2e}")

assert abs(UB - ref_obj) < 1e-4, "¡Las soluciones difieren más de lo esperado!"
print("  ✓ Las soluciones coinciden dentro de tolerancia 1e-4")

# Gráfico de convergencia
fig, ax = plt.subplots(figsize=(9, 4))
iters = list(range(1, len(history_LB) + 1))
ax.plot(iters, history_LB, color="#254F83", lw=2, label="LB")
ub_iters = [iters[i] for i, v in enumerate(history_UB) if v is not None]
ub_vals = [v for v in history_UB if v is not None]
ax.plot(ub_iters, ub_vals, color="#CC3030", lw=2, linestyle="--", label="UB")
ax.axhline(ref_obj, color="gray", lw=1, linestyle=":", label=f"Óptimo ({ref_obj:.4f})")
ax.set_xlabel("Iteración de Benders")
ax.set_ylabel("Valor de la cota")
ax.set_title("Convergencia de la Descomposición de Benders — LP Bloque")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("benders_01_convergencia.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: benders_01_convergencia.png")
plt.show()
