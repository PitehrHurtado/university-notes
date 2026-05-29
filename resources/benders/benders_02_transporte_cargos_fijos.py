"""
Descomposición de Benders — Ejemplo 2: Fixed-Charge Transportation Problem (FCTP)
==================================================================================
Variables complicantes: y_{ij} in {0,1} (abrir ruta i→j), costo fijo f_{ij}.
Variables de recourse:  x_{ij} >= 0 (flujo en ruta i→j), costo variable c_{ij}.

Subproblema (y fijo): LP de transporte clásico.
Master:               MILP con variables binarias y_{ij} y surrogate eta.

Los cortes de optimalidad se construyen con los precios sombra del LP de transporte.
Los cortes de factibilidad usan el rayo de Farkas (FarkasDual de Gurobi).

Estructura:
    1. DATOS
    2. PROBLEMA ORIGINAL MIP (Gurobi — referencia)
    3. FUNCIONES AUXILIARES (Master, Subproblema)
    4. LOOP PRINCIPAL DE BENDERS
    5. ANÁLISIS Y COMPARACIÓN
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(123)

n_origins = 4      # orígenes (i)
n_dests = 5        # destinos (j)
I = range(n_origins)
J = range(n_dests)

# Oferta y demanda (balanceado: sum supply = sum demand)
supply = rng.integers(20, 50, size=n_origins).astype(float)
demand = rng.integers(15, 40, size=n_dests).astype(float)
# balancear
demand = demand / demand.sum() * supply.sum()

# Costos variables c[i,j] y fijos f[i,j]
var_cost = rng.integers(1, 15, size=(n_origins, n_dests)).astype(float)
fix_cost = rng.integers(10, 35, size=(n_origins, n_dests)).astype(float)

# Big-M para la restricción x_{ij} <= M y_{ij}
BIG_M = float(supply.max())

print("=" * 65)
print("EJEMPLO BENDERS 2: FIXED-CHARGE TRANSPORTATION (FCTP)")
print("=" * 65)
print(f"  Orígenes: {n_origins}  |  Destinos: {n_dests}")
print(f"  Oferta: {supply}")
print(f"  Demanda: {demand.round(1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEMA ORIGINAL MIP (Gurobi — referencia)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Solución MIP óptima (Gurobi) ---")
m_mip = gp.Model("fctp_mip")
m_mip.setParam("OutputFlag", 0)

y_mip = m_mip.addVars(I, J, vtype=GRB.BINARY, name="y")
x_mip = m_mip.addVars(I, J, lb=0, name="x")

m_mip.setObjective(
    gp.quicksum(var_cost[i, j] * x_mip[i, j] + fix_cost[i, j] * y_mip[i, j]
                for i in I for j in J),
    GRB.MINIMIZE
)
# oferta
for i in I:
    m_mip.addConstr(gp.quicksum(x_mip[i, j] for j in J) <= supply[i], name=f"sup_{i}")
# demanda
for j in J:
    m_mip.addConstr(gp.quicksum(x_mip[i, j] for i in I) >= demand[j], name=f"dem_{j}")
# big-M linking
for i in I:
    for j in J:
        m_mip.addConstr(x_mip[i, j] <= BIG_M * y_mip[i, j], name=f"link_{i}_{j}")

m_mip.optimize()
ref_obj = m_mip.ObjVal
y_ref = np.array([[y_mip[i, j].X for j in J] for i in I])
x_ref = np.array([[x_mip[i, j].X for j in J] for i in I])
print(f"  Valor óptimo MIP: {ref_obj:.4f}")
print(f"  Rutas abiertas: {[(i, j) for i in I for j in J if y_ref[i, j] > 0.5]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def solve_master(opt_cuts: list, feas_cuts: list) -> tuple:
    """
    Restricted Master Problem (MILP):
        min  sum_{ij} f_{ij} y_{ij} + eta
        s.a. [cortes de optimalidad]
             [cortes de factibilidad]
             y_{ij} in {0,1}
             eta >= -1e6

    Retorna: (y_val np.ndarray, eta_val float, obj_val float)
    """
    m = gp.Model("rmp_fctp")
    m.setParam("OutputFlag", 0)

    y = m.addVars(I, J, vtype=GRB.BINARY, name="y")
    eta = m.addVar(lb=-1e6, name="eta")

    m.setObjective(
        gp.quicksum(fix_cost[i, j] * y[i, j] for i in I for j in J) + eta,
        GRB.MINIMIZE
    )

    # cortes de optimalidad: eta >= u_sup^T s - u_dem^T d + sum_{ij}(u_dem_j - u_sup_i)*M*y_{ij}
    for (u_sup, u_dem), _ in opt_cuts:
        # Q(y) >= u_sup^T supply - u_dem^T demand + sum_ij (u_dem_j - u_sup_i) * M * y_ij
        # (u_sup <= 0, u_dem >= 0 por convención del LP de transporte min)
        rhs_const = (gp.quicksum(u_sup[i] * supply[i] for i in I) +
                     gp.quicksum(u_dem[j] * demand[j] for j in J))
        y_coef = gp.quicksum(
            (u_dem[j] - u_sup[i]) * BIG_M * y[i, j]
            for i in I for j in J
        )
        m.addConstr(eta >= rhs_const + y_coef)

    # cortes de factibilidad: r_sup^T supply + r_dem^T demand + sum_{ij}(r_dem_j - r_sup_i)*M*y_{ij} <= 0
    for (r_sup, r_dem), _ in feas_cuts:
        rhs_const = (gp.quicksum(r_sup[i] * supply[i] for i in I) +
                     gp.quicksum(r_dem[j] * demand[j] for j in J))
        y_coef = gp.quicksum(
            (r_dem[j] - r_sup[i]) * BIG_M * y[i, j]
            for i in I for j in J
        )
        m.addConstr(rhs_const + y_coef <= 0)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"RMP infactible o ilimitado (status={m.Status})")

    y_val = np.array([[y[i, j].X for j in J] for i in I])
    return y_val, eta.X, m.ObjVal


def solve_transport_subproblem(y_fixed: np.ndarray) -> dict:
    """
    LP de transporte con capacidad de ruta ligada a y_fixed:
        min  sum_{ij} c_{ij} x_{ij}
        s.a. sum_j x_{ij} <= supply[i]          (oferta)
             sum_i x_{ij} >= demand[j]           (demanda)
             x_{ij} <= M * y_fixed[i,j]          (capacidad de ruta)
             x_{ij} >= 0

    Retorna dict con claves:
        status, obj, u_sup (dual oferta), u_dem (dual demanda),
        farkas_sup, farkas_dem
    """
    m = gp.Model("transport_sp")
    m.setParam("OutputFlag", 0)
    m.setParam("InfUnbdInfo", 1)

    x = m.addVars(I, J, lb=0, name="x")

    m.setObjective(
        gp.quicksum(var_cost[i, j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE
    )

    sup_constrs = []
    dem_constrs = []

    for i in I:
        c = m.addConstr(
            gp.quicksum(x[i, j] for j in J) <= supply[i],
            name=f"sup_{i}"
        )
        sup_constrs.append(c)

    for j in J:
        c = m.addConstr(
            gp.quicksum(x[i, j] for i in I) >= demand[j],
            name=f"dem_{j}"
        )
        dem_constrs.append(c)

    # capacidad de ruta
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= BIG_M * y_fixed[i, j], name=f"cap_{i}_{j}")

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        u_sup = np.array([c.Pi for c in sup_constrs])   # <= 0 (cota sup activa)
        u_dem = np.array([c.Pi for c in dem_constrs])   # >= 0 (cota inf activa)
        return {
            'status': 'optimal',
            'obj': m.ObjVal,
            'u_sup': u_sup, 'u_dem': u_dem,
            'farkas_sup': None, 'farkas_dem': None
        }
    elif m.Status == GRB.INFEASIBLE:
        r_sup = np.array([c.FarkasDual for c in sup_constrs])
        r_dem = np.array([c.FarkasDual for c in dem_constrs])
        return {
            'status': 'infeasible',
            'obj': None,
            'u_sup': None, 'u_dem': None,
            'farkas_sup': r_sup, 'farkas_dem': r_dem
        }
    else:
        raise RuntimeError(f"SP status inesperado: {m.Status}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOOP PRINCIPAL DE BENDERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITER = 60
EPSILON = 1e-4

opt_cuts = []   # lista de ((u_sup, u_dem), y_k)
feas_cuts = []  # lista de ((r_sup, r_dem), y_k)

LB = -np.inf
UB = np.inf
best_y_benders = None

history_LB = []
history_UB = []
cut_types = []

print("\n--- Algoritmo de Benders (FCTP) ---")
print(f"{'Iter':>4} | {'LB':>12} | {'UB':>12} | {'gap%':>8} | {'corte'}")
print("-" * 55)

for k in range(1, MAX_ITER + 1):
    # Paso 1: resolver RMP
    y_k, eta_k, rmp_obj = solve_master(opt_cuts, feas_cuts)
    LB = rmp_obj

    # Paso 2: resolver LP de transporte con y^k
    sp = solve_transport_subproblem(y_k)

    if sp['status'] == 'optimal':
        u_sup, u_dem = sp['u_sup'], sp['u_dem']
        ub_candidate = np.sum(fix_cost * y_k) + sp['obj']
        if ub_candidate < UB:
            UB = ub_candidate
            best_y_benders = y_k.copy()

        opt_cuts.append(((u_sup, u_dem), y_k.copy()))
        cut_type = "optimalidad"
    else:
        r_sup, r_dem = sp['farkas_sup'], sp['farkas_dem']
        feas_cuts.append(((r_sup, r_dem), y_k.copy()))
        cut_type = "factibilidad"

    cut_types.append(cut_type)
    history_LB.append(LB)
    history_UB.append(UB if UB < np.inf else None)

    gap = (UB - LB) / (abs(LB) + 1e-9) if UB < np.inf else np.inf
    ub_str = f"{UB:12.4f}" if UB < np.inf else f"{'∞':>12}"
    print(f"{k:>4} | {LB:12.4f} | {ub_str} | {gap*100:7.3f}% | {cut_type}")

    if gap < EPSILON:
        print(f"\n  Convergió en iteración {k} (gap = {gap:.2e})")
        break

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS Y COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VERIFICACIÓN Y COMPARACIÓN")
print("=" * 65)
print(f"  Solución Benders (UB)  : {UB:.4f}")
print(f"  Solución Gurobi MIP    : {ref_obj:.4f}")
print(f"  Diferencia             : {abs(UB - ref_obj):.2e}")
print(f"  Cortes de optimalidad  : {sum(1 for t in cut_types if t == 'optimalidad')}")
print(f"  Cortes de factibilidad : {sum(1 for t in cut_types if t == 'factibilidad')}")

if best_y_benders is not None:
    print(f"  Rutas abiertas (Benders): "
          f"{[(i, j) for i in I for j in J if best_y_benders[i, j] > 0.5]}")

# Gráfico de convergencia
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

iters = list(range(1, len(history_LB) + 1))
axes[0].plot(iters, history_LB, color="#254F83", lw=2, label="LB (RMP)")
ub_iters_plot = [iters[i] for i, v in enumerate(history_UB) if v is not None]
ub_vals_plot = [v for v in history_UB if v is not None]
axes[0].plot(ub_iters_plot, ub_vals_plot, color="#CC3030", lw=2, linestyle="--", label="UB")
axes[0].axhline(ref_obj, color="gray", lw=1.2, linestyle=":", label=f"MIP óptimo ({ref_obj:.1f})")

# marcar iteraciones de cortes de factibilidad
for k_idx, ct in enumerate(cut_types):
    if ct == "factibilidad":
        axes[0].axvline(k_idx + 1, color="orange", alpha=0.4, lw=1)

axes[0].set_xlabel("Iteración de Benders")
axes[0].set_ylabel("Valor de la cota")
axes[0].set_title("Convergencia Benders — FCTP\n(líneas naranjas = cortes de factibilidad)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Heatmap de costos fijos
im = axes[1].imshow(fix_cost, cmap="Blues", aspect="auto")
axes[1].set_xlabel("Destino j")
axes[1].set_ylabel("Origen i")
axes[1].set_title("Costos fijos de apertura f[i,j]")
for i in I:
    for j in J:
        axes[1].text(j, i, f"{fix_cost[i,j]:.0f}", ha="center", va="center",
                     fontsize=7, color="black")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("benders_02_fctp_convergencia.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: benders_02_fctp_convergencia.png")
plt.show()
