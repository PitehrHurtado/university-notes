"""
Descomposición de Benders — Ejemplo 3: Two-Stage Stochastic MILP
================================================================
Problema: expansión de capacidad bajo incertidumbre en la demanda.

Primera etapa  (aquí y ahora): decidir qué instalaciones construir.
    z[i]  in {0,1}: construir instalación i, costo fijo f[i]
    cap[i]: capacidad de la instalación i (fija si z[i]=1)

Segunda etapa  (recourse, por escenario s): asignar demanda a instalaciones.
    x[i,j,s] >= 0: unidades entregadas desde instalación i a mercado j en escenario s

El subproblema de segunda etapa para escenario s es un LP de asignación.
Se implementa con CALLBACKS de Gurobi (lazy constraints).

Estructura:
    1. DATOS
    2. FORMULACIÓN Y EQUIVALENTE DETERMINÍSTICO (Gurobi, referencia)
    3. CLASE CALLBACK DE BENDERS
    4. PROBLEMA MAESTRO CON CALLBACK
    5. ANÁLISIS: VSS, EVPI (aproximado)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(99)

n_fac = 3      # instalaciones
n_mkt = 4      # mercados
n_scen = 8     # escenarios

I = range(n_fac)
J = range(n_mkt)
S = range(n_scen)

# Costos fijos de construcción f[i]
fix_cost = np.array([80., 120., 100.])
# Capacidad instalada si z[i]=1
cap = np.array([60., 90., 70.])
# Costo de transporte c[i,j] (por unidad)
unit_cost = np.array([
    [3., 5., 4., 6.],
    [2., 4., 6., 3.],
    [5., 3., 2., 4.],
])  # n_fac x n_mkt

# Probabilidades de escenario (uniformes)
prob = np.ones(n_scen) / n_scen

# Demandas por escenario d[j,s]
base_demand = np.array([30., 40., 35., 25.])
demand_scenarios = np.outer(base_demand, np.ones(n_scen)) + rng.normal(0, 8, size=(n_mkt, n_scen))
demand_scenarios = np.clip(demand_scenarios, 5, None)  # demanda no negativa

print("=" * 65)
print("EJEMPLO BENDERS 3: TWO-STAGE STOCHASTIC MILP (CALLBACK)")
print("=" * 65)
print(f"  Instalaciones: {n_fac}  |  Mercados: {n_mkt}  |  Escenarios: {n_scen}")
print(f"  Costos fijos f: {fix_cost}")
print(f"  Capacidades cap: {cap}")
print(f"  Demanda media por mercado: {demand_scenarios.mean(axis=1).round(1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EQUIVALENTE DETERMINÍSTICO (referencia Gurobi)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Equivalente determinístico (Gurobi) ---")
m_det = gp.Model("deterministic_eq")
m_det.setParam("OutputFlag", 0)

z_det = m_det.addVars(I, vtype=GRB.BINARY, name="z")
x_det = m_det.addVars(I, J, S, lb=0, name="x")

# Objetivo: costos fijos + E[costos de transporte]
m_det.setObjective(
    gp.quicksum(fix_cost[i] * z_det[i] for i in I) +
    gp.quicksum(
        prob[s] * unit_cost[i, j] * x_det[i, j, s]
        for i in I for j in J for s in S
    ),
    GRB.MINIMIZE
)

# Capacidad (si instalación abierta)
for i in I:
    for s in S:
        m_det.addConstr(
            gp.quicksum(x_det[i, j, s] for j in J) <= cap[i] * z_det[i],
            name=f"cap_{i}_{s}"
        )

# Satisfacción de demanda
for j in J:
    for s in S:
        m_det.addConstr(
            gp.quicksum(x_det[i, j, s] for i in I) >= demand_scenarios[j, s],
            name=f"dem_{j}_{s}"
        )

m_det.optimize()
ref_obj = m_det.ObjVal
z_det_sol = np.array([z_det[i].X for i in I])
print(f"  Costo total equivalente determinístico: {ref_obj:.4f}")
print(f"  Instalaciones construidas: {[i for i in I if z_det_sol[i] > 0.5]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASE CALLBACK DE BENDERS
# ─────────────────────────────────────────────────────────────────────────────

class BendersCallback:
    """
    Callback de Gurobi para agregar cortes de Benders como lazy constraints.

    En cada nodo entero (MIPSOL), resuelve el subproblema LP para cada
    escenario s. Si el subproblema es infactible, agrega corte de factibilidad.
    Si la variable surrogate eta[s] viola el corte de optimalidad, lo agrega.
    """

    def __init__(self, z_vars: dict, eta_vars: dict, data: dict):
        self.z_vars = z_vars
        self.eta_vars = eta_vars
        self.data = data
        self.n_opt_cuts = 0
        self.n_feas_cuts = 0
        self.iter_log = []  # (z_val, eta_val, sp_obj_por_escenario)

    def __call__(self):
        if self.where == GRB.Callback.MIPSOL:
            # Extraer solución entera actual
            z_val = np.array([self.cbGetSolution(self.z_vars[i]) for i in I])
            eta_val = np.array([self.cbGetSolution(self.eta_vars[s]) for s in S])

            total_sp_obj = 0.0
            cuts_added = False

            for s in S:
                sp_result = self._solve_subproblem(z_val, s)

                if sp_result['status'] == 'infeasible':
                    # Corte de factibilidad: r^T (b - B z) <= 0
                    # r_cap[i]: dual de Farkas de restricción de capacidad instalación i
                    # r_dem[j]: dual de Farkas de restricción de demanda mercado j
                    r_cap = sp_result['farkas_cap']
                    r_dem = sp_result['farkas_dem']

                    lhs = gp.LinExpr()
                    # -sum_i r_cap[i] * cap[i] * z[i] - sum_j r_dem[j] * d[j,s] <= 0
                    # => sum_i r_cap[i] * cap[i] * z[i] >= sum_j r_dem[j] * d[j,s]
                    rhs = float(sum(r_dem[j] * demand_scenarios[j, s] for j in J))
                    for i in I:
                        lhs += r_cap[i] * cap[i] * self.z_vars[i]
                    self.cbCut(lhs, GRB.GREATER_EQUAL, rhs)
                    self.n_feas_cuts += 1
                    cuts_added = True

                else:
                    # Corte de optimalidad: eta[s] >= (u_cap)^T (-cap*z) + (u_dem)^T d[s]
                    u_cap = sp_result['u_cap']   # dual de cap (<=0 típicamente)
                    u_dem = sp_result['u_dem']   # dual de dem (>=0 típicamente)
                    sp_obj = sp_result['obj']
                    total_sp_obj += prob[s] * sp_obj

                    # Corte: eta[s] >= sum_i u_cap[i]*(-cap[i])*z[i] + sum_j u_dem[j]*d[j,s]
                    # = Q_s(z) = max_u { u_cap^T(-cap*z) + u_dem^T d_s }
                    rhs = float(sum(u_dem[j] * demand_scenarios[j, s] for j in J))
                    lhs = gp.LinExpr()
                    lhs += self.eta_vars[s]
                    for i in I:
                        lhs -= u_cap[i] * (-cap[i]) * self.z_vars[i]
                    # equivalente: eta[s] + sum_i u_cap[i]*cap[i]*z[i] >= sum_j u_dem[j]*d[j,s]

                    # verificar si el corte está violado
                    cut_val = float(sum(u_dem[j] * demand_scenarios[j, s] for j in J) +
                                    sum(u_cap[i] * (-cap[i]) * z_val[i] for i in I))
                    if cut_val > eta_val[s] + 1e-4:
                        expr = self.eta_vars[s] - gp.quicksum(
                            u_cap[i] * (-cap[i]) * self.z_vars[i] for i in I
                        )
                        self.cbCut(expr, GRB.GREATER_EQUAL, rhs)
                        self.n_opt_cuts += 1
                        cuts_added = True

            self.iter_log.append({
                'z': z_val.copy(),
                'eta': eta_val.copy(),
                'n_opt': self.n_opt_cuts,
                'n_feas': self.n_feas_cuts,
            })

    def _solve_subproblem(self, z_val: np.ndarray, s: int) -> dict:
        """
        LP de asignación para escenario s con z fijado:
            min  sum_{ij} c_{ij} x_{ij}
            s.a. sum_j x_{ij} <= cap[i] * z[i]   (capacidad)
                 sum_i x_{ij} >= d[j,s]            (demanda)
                 x_{ij} >= 0
        """
        m = gp.Model(f"sp_s{s}")
        m.setParam("OutputFlag", 0)
        m.setParam("InfUnbdInfo", 1)

        x = m.addVars(I, J, lb=0, name="x")

        m.setObjective(
            gp.quicksum(unit_cost[i, j] * x[i, j] for i in I for j in J),
            GRB.MINIMIZE
        )

        cap_constrs = []
        for i in I:
            c = m.addConstr(
                gp.quicksum(x[i, j] for j in J) <= cap[i] * z_val[i],
                name=f"cap_{i}"
            )
            cap_constrs.append(c)

        dem_constrs = []
        for j in J:
            c = m.addConstr(
                gp.quicksum(x[i, j] for i in I) >= demand_scenarios[j, s],
                name=f"dem_{j}"
            )
            dem_constrs.append(c)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            u_cap = np.array([c.Pi for c in cap_constrs])
            u_dem = np.array([c.Pi for c in dem_constrs])
            return {
                'status': 'optimal',
                'obj': m.ObjVal,
                'u_cap': u_cap,
                'u_dem': u_dem,
                'farkas_cap': None,
                'farkas_dem': None,
            }
        else:  # INFEASIBLE
            farkas_cap = np.array([c.FarkasDual for c in cap_constrs])
            farkas_dem = np.array([c.FarkasDual for c in dem_constrs])
            return {
                'status': 'infeasible',
                'obj': np.inf,
                'u_cap': None,
                'u_dem': None,
                'farkas_cap': farkas_cap,
                'farkas_dem': farkas_dem,
            }

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROBLEMA MAESTRO CON CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Benders con callback de Gurobi ---")
master = gp.Model("benders_master")
master.setParam("OutputFlag", 1)
master.setParam("LazyConstraints", 1)  # OBLIGATORIO para cbCut
master.setParam("TimeLimit", 120)
master.setParam("MIPGap", 1e-4)

# Variables de primera etapa
z_vars = {i: master.addVar(vtype=GRB.BINARY, name=f"z_{i}") for i in I}
# Variables surrogate de recourse por escenario
eta_vars = {s: master.addVar(lb=-1e6, name=f"eta_{s}") for s in S}

# Objetivo del maestro
master.setObjective(
    gp.quicksum(fix_cost[i] * z_vars[i] for i in I) +
    gp.quicksum(prob[s] * eta_vars[s] for s in S),
    GRB.MINIMIZE
)

# Sin más restricciones iniciales en el maestro
# (los cortes se generan dinámicamente en el callback)

cb = BendersCallback(z_vars, eta_vars, {})
master.optimize(cb)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS: RESULTADOS, VSS, EVPI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS")
print("=" * 65)

if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    benders_obj = master.ObjVal
    z_benders = np.array([z_vars[i].X for i in I])
    print(f"  Costo Benders          : {benders_obj:.4f}")
    print(f"  Instalaciones construidas: {[i for i in I if z_benders[i] > 0.5]}")
    print(f"  Costo Gurobi directo   : {ref_obj:.4f}")
    print(f"  Diferencia             : {abs(benders_obj - ref_obj):.2e}")
    print(f"\n  Cortes de optimalidad  : {cb.n_opt_cuts}")
    print(f"  Cortes de factibilidad : {cb.n_feas_cuts}")
else:
    benders_obj = np.inf
    z_benders = np.zeros(n_fac)
    print(f"  Maestro no convergió (status={master.Status})")

# ── Value of Stochastic Solution (VSS) ───────────────────────────────────────
# EEV: resolver deterministic con demanda esperada y luego evaluar en escenarios
print("\n--- Value of Stochastic Solution (VSS) ---")
mean_demand = demand_scenarios.mean(axis=1)

m_eev = gp.Model("eev")
m_eev.setParam("OutputFlag", 0)
z_eev = m_eev.addVars(I, vtype=GRB.BINARY, name="z")
x_eev = m_eev.addVars(I, J, lb=0, name="x")
m_eev.setObjective(
    gp.quicksum(fix_cost[i] * z_eev[i] for i in I) +
    gp.quicksum(unit_cost[i, j] * x_eev[i, j] for i in I for j in J),
    GRB.MINIMIZE
)
for i in I:
    m_eev.addConstr(gp.quicksum(x_eev[i, j] for j in J) <= cap[i] * z_eev[i])
for j in J:
    m_eev.addConstr(gp.quicksum(x_eev[i, j] for i in I) >= mean_demand[j])
m_eev.optimize()
z_eev_sol = np.array([z_eev[i].X for i in I])

# Evaluar z_eev_sol en todos los escenarios (EEV)
eev_total = float(sum(fix_cost[i] * z_eev_sol[i] for i in I))
for s in S:
    sp = cb._solve_subproblem(z_eev_sol, s)
    if sp['status'] == 'optimal':
        eev_total += prob[s] * sp['obj']
    else:
        eev_total += prob[s] * 1e6  # penalidad por infactibilidad

print(f"  Costo EEV (esperanza de valor esperado): {eev_total:.4f}")
print(f"  Costo RP  (recourse problem = Benders) : {benders_obj:.4f}")
vss = eev_total - benders_obj
print(f"  VSS = EEV - RP                          : {vss:.4f}  ({vss/benders_obj*100:.2f}%)")

# ── EVPI (aproximado via wait-and-see) ───────────────────────────────────────
print("\n--- EVPI (aprox. via wait-and-see) ---")
ws_total = 0.0
for s in S:
    m_ws = gp.Model(f"ws_{s}")
    m_ws.setParam("OutputFlag", 0)
    z_ws = m_ws.addVars(I, vtype=GRB.BINARY, name="z")
    x_ws = m_ws.addVars(I, J, lb=0, name="x")
    m_ws.setObjective(
        gp.quicksum(fix_cost[i] * z_ws[i] for i in I) +
        gp.quicksum(unit_cost[i, j] * x_ws[i, j] for i in I for j in J),
        GRB.MINIMIZE
    )
    for i in I:
        m_ws.addConstr(gp.quicksum(x_ws[i, j] for j in J) <= cap[i] * z_ws[i])
    for j in J:
        m_ws.addConstr(gp.quicksum(x_ws[i, j] for i in I) >= demand_scenarios[j, s])
    m_ws.optimize()
    ws_total += prob[s] * m_ws.ObjVal

evpi = benders_obj - ws_total
print(f"  WS (wait-and-see esperado)  : {ws_total:.4f}")
print(f"  EVPI = RP - WS              : {evpi:.4f}  ({evpi/benders_obj*100:.2f}%)")

# ── Gráfico ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Barras de costos: inversión vs recourse esperado
inv_cost = float(sum(fix_cost[i] * z_benders[i] for i in I)) if benders_obj < np.inf else 0
recourse_cost = benders_obj - inv_cost if benders_obj < np.inf else 0
categories = ["Inversión", "Recourse E[Q(z,ξ)]"]
values = [inv_cost, recourse_cost]
bars = axes[0].bar(categories, values, color=["#254F83", "#CC3030"], alpha=0.8, edgecolor="white")
axes[0].set_ylabel("Costo ($)")
axes[0].set_title("Descomposición del costo óptimo\n(Solución Benders)")
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"${val:.1f}", ha="center", fontsize=10, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

# Comparación VSS / EVPI
labels = ["WS\n(wait-and-see)", "RP\n(Benders)", "EEV\n(det.media)"]
costs = [ws_total, benders_obj, eev_total]
colors_bar = ["#4A7A2C", "#254F83", "#CC3030"]
bars2 = axes[1].bar(labels, costs, color=colors_bar, alpha=0.8, edgecolor="white")
axes[1].set_ylabel("Costo esperado ($)")
axes[1].set_title("Comparación: WS, RP, EEV\n(VSS = EEV−RP,  EVPI = RP−WS)")
for bar, val in zip(bars2, costs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${val:.1f}", ha="center", fontsize=9, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("benders_03_estocastico_resultados.png", dpi=120, bbox_inches="tight")
print("\n  Gráfico guardado: benders_03_estocastico_resultados.png")
plt.show()
