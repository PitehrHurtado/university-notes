"""
part_A_Hurtado.py

Name: Pitehr Hurtado-Cayo
=================
Part A of the Coding Exam.
    Content:
        A.1  Build and solve the base MILP.
        A.2  Holding-cost sensitivity (alpha param).
        A.3  Overtime-cost sensitivity (beta param).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# Set script directory for saving figures and ouptuts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 1. SETS
# ===========================================================================
K = ['A', 'B', 'C']
M = ['F1', 'F2']
J = ['DC1', 'DC2']
T = [1, 2, 3, 4]

# ===========================================================================
# 2. PARAMETERS
# ===========================================================================

# D[j,k,t] = demand of product k at DC j in period t
D = {
    ('DC1','A',1):28, ('DC1','A',2):27, ('DC1','A',3):33, ('DC1','A',4):30,
    ('DC1','B',1):17, ('DC1','B',2):20, ('DC1','B',3):26, ('DC1','B',4):22,
    ('DC1','C',1):11, ('DC1','C',2):17, ('DC1','C',3):14, ('DC1','C',4):17,
    ('DC2','A',1):27, ('DC2','A',2):30, ('DC2','A',3):28, ('DC2','A',4):33,
    ('DC2','B',1):19, ('DC2','B',2):19, ('DC2','B',3):26, ('DC2','B',4):24,
    ('DC2','C',1):13, ('DC2','C',2):19, ('DC2','C',3):17, ('DC2','C',4):18,
}

# parameteres auxiliary for cost calculations
c0_m  = {'F1': 8,  'F2': 10}
c0_k  = {'A': 0,  'B': 1,  'C': 2}
# =========================================================================

# CP[m,k,t] = production cost per unit of product k at furnace m in period t
CP    = {(m, k, t): c0_m[m] + c0_k[k] for m in M for k in K for t in T}

# CT[j,k,t] = transportation cost per unit of product k from furnace to DC j in period t
CT    = {(j, k, t): (1 if j == 'DC1' else 2) for j in J for k in K for t in T}

# CH[k] = holding-cost per unit of product k at end of period t
CH_base  = {'A': 1.0, 'B': 1.2, 'C': 0.8}

# CB[k] = backorder-cost per unit of product k at end of period t
CB       = {'A': 4,   'B': 5,   'C': 3}

# a_mach[m,k] = machine-hour per unit consumed for one unit of product k at furnace m
a_mach   = {(m, k): v for m in M for k, v in zip(K, [0.5, 0.7, 0.4])}

# H_reg[m,t] = regular machine hours available at furnace m in period t
H_reg    = {(m, t): 50  for m in M for t in T}

# H_ov[m,t] = overtime machine hours available at furnace m in period t
H_ov     = {(m, t): 20  for m in M for t in T}

# CF[m,t] = fixed cost for using furnace m in period t
CF       = {(m, t): 200 for m in M for t in T}

# cov[m,t] = overtime cost per extra-hour at furnace m in period t
cov_base = {(m, t): 15  for m in M for t in T}

# initial inventory
I0      = {'A': 0, 'B': 0, 'C': 0}

# S_cap[k] = inventory capacity for product k at end of period t
S_cap   = {'A': 200, 'B': 200, 'C': 200}

# service level requirement: rho[k] = minimum fraction of demand to be met on time for product k in each period
rho     = {'A': 0.60, 'B': 0.60, 'C': 0.60}

# epsilon = production smoothing parameter: max allowed relative change in production quantity of product k between consecutive periods
epsilon = 0.25

# ===========================================================================
# 3. MODEL BUILDER
# ===========================================================================

def build_and_solve(alpha=1.0, beta=1.0, output_flag=False):
    """
    Params:
    alpha : multiplier on holding costs CH_k -> alpha * CH_k
    beta  : multiplier on overtime costs cov -> beta * cov

    Considering default parameters, alpha=1.0 and beta=1.0 correspond to the base model.
    
    Returns (model, vars_dict) or (None, None) if not optimal.
    """
    mdl = gp.Model("production_planning_problem")
    mdl.setParam('OutputFlag', 1 if output_flag else 0)  # Suppress Gurobi output

    # Adjusted costs based on sensitivity parameters
    CH  = {k: alpha * CH_base[k] for k in K}
    cov = {(m, t): beta * cov_base[m, t] for m in M for t in T}

    # ===================================================================================
    # Variables
    # ===================================================================================
    # x[m,k,t] = unit of k produced on furnace m in period t
    x   = mdl.addVars(M, K, T, vtype=GRB.CONTINUOUS, lb=0, name='x')

    # y[j,k,t] = unit of k delivered to DC j in period t
    y   = mdl.addVars(J, K, T, vtype=GRB.CONTINUOUS, lb=0, name='y')

    # inv[k,t] = physical inventory of k at end of period t
    # note: I can limit inventory to 200 units (S_cap).
    # Contraint (6) will enforce this, but adding an upper bound here can help the solver.
    inv = mdl.addVars(K, T, vtype=GRB.CONTINUOUS, lb=0, ub=200, name='inventory')

    # bkd[k,t] = cumulativebackorder of k at end of period t
    bkd = mdl.addVars(K, T, vtype=GRB.CONTINUOUS, lb=0, name='backorder')
    
    # ot[m,t] = overtime hours used on furnace m in period t
    ot  = mdl.addVars(M, T, vtype=GRB.CONTINUOUS, lb=0, name='overtime')

    # z[m,t] = binary variable indicating if furnace m is used in period t
    z   = mdl.addVars(M, T, vtype=GRB.BINARY, name='furnace_used')

    # Objective (1)
    mdl.setObjective(
        quicksum(CP[m,k,t]  * x[m,k,t]  for m in M for k in K for t in T) +
        quicksum(CT[j,k,t]  * y[j,k,t]  for j in J for k in K for t in T) +
        quicksum(CF[m,t]    * z[m,t]     for m in M for t in T) +
        quicksum(cov[m,t]   * ot[m,t]    for m in M for t in T) +
        quicksum(CH[k]      * inv[k,t]   for k in K for t in T) +
        quicksum(CB[k]      * bkd[k,t]   for k in K for t in T),
        GRB.MINIMIZE
    )

    # (2) Inventory balance
    for k in K:
        for t in T:
            inv_prev = I0[k] if t == 1 else inv[k, t-1]
            mdl.addConstr(
                inv[k,t] == inv_prev
                           + quicksum(x[m,k,t] for m in M)
                           - quicksum(y[j,k,t] for j in J),
                name=f'inv_bal_{k}_{t}'
            )

    # (3) Backorder balance
    for k in K:
        for t in T:
            bkd_prev = 0 if t == 1 else bkd[k, t-1]
            mdl.addConstr(
                quicksum(y[j,k,t] for j in J) + bkd[k,t]
                == quicksum(D[j,k,t] for j in J) + bkd_prev,
                name=f'backorder_bal_{k}_{t}'
            )

    # (4) Final backorders = 0
    for k in K:
        mdl.addConstr(bkd[k, len(T)] == 0, name=f'backorder_final_{k}')

    # (5) Service level
    for k in K:
        for t in T:
            mdl.addConstr(
                quicksum(y[j,k,t] for j in J)
                >= rho[k] * quicksum(D[j,k,t] for j in J),
                name=f'service_{k}_{t}'
            )

    # (6) Inventory cap
    # Note: I already set an UB of 200 in inv[k,t] when defining the variable, so this constraint is redundant
    # for k in K:
    #     for t in T:
    #         mdl.addConstr(inv[k,t] <= S_cap[k], name=f'inv_cap_{k}_{t}')

    # (7) Machine hours
    for m in M:
        for t in T:
            mdl.addConstr(
                quicksum(a_mach[m,k] * x[m,k,t] for k in K)
                <= H_reg[m,t] * z[m,t] + ot[m,t],
                name=f'hours_used_{m}_{t}'
            )

    # (8) Overtime limit
    for m in M:
        for t in T:
            mdl.addConstr(
                ot[m,t] <= H_ov[m,t] * z[m,t], 
                name=f'overtime_lim_{m}_{t}'
            )

    # (9)-(10) Production smoothing
    for k in K:
        for t in T:
            if t >= 2:
                X_kt = quicksum(x[m,k,t]  for m in M)
                X_kt_1 = quicksum(x[m,k,t-1] for m in M)

                mdl.addConstr(X_kt <= (1 + epsilon) * X_kt_1, name=f'smooth_up_{k}_{t}')
                mdl.addConstr(X_kt >= (1 - epsilon) * X_kt_1, name=f'smooth_down_{k}_{t}')

    mdl.optimize()

    if mdl.Status != GRB.OPTIMAL:
        return None, None

    return mdl, {'x': x, 'y': y, 'inv': inv, 'bkd': bkd, 'ot': ot, 'z': z}


# ===========================================================================
# 4. A.1 — SOLVE AND REPORT
# ===========================================================================

print("\n" + "="*70)
print("  A.1 — BASE MODEL: OPTIMAL SOLUTION")
print("="*70)

mdl, vs = build_and_solve(alpha=1.0, beta=1.0, output_flag=True)
if mdl is None:
    raise SystemExit("Base model infeasible, sad!")

x_v, y_v, inv_v, bkd_v, ot_v, z_v = (
    vs['x'], vs['y'], vs['inv'], vs['bkd'], vs['ot'], vs['z']
)
# ===========================================================================
# i) Optimal objective value
# ===========================================================================
print(f"\nOptimal objective value: {mdl.ObjVal:,.4f}\n")

# ===========================================================================
# ii) Actual Demand / Optimal production / delivery / inventory / backorder table
# ===========================================================================
rows = []
for k in K:
    for t in T:
        D_kt = sum(D[j,k,t]   for j in J)
        X_kt = sum(x_v[m,k,t].X for m in M)
        Y_kt = sum(y_v[j,k,t].X for j in J)
        I_kt = inv_v[k,t].X
        B_kt = bkd_v[k,t].X
        rows.append({
            'Product': k, 't': t,
            'D_kt': D_kt,
            'X_kt': round(X_kt, 4),
            'Y_kt': round(Y_kt, 4),
            'I_kt': round(I_kt, 4),
            'B_kt': round(B_kt, 4),
        })
df_prod = pd.DataFrame(rows)
print("Demand / Production / Delivery / Inventory / Backorder Table:")
print(df_prod.to_string(index=False))

# ===========================================================================
# iii) Furnace table (z[m,t], hours used, overtime used)
# ===========================================================================
rows2 = []
for m in M:
    for t in T:
        z_val      = int(round(z_v[m,t].X))
        ot_val     = ot_v[m,t].X
        hours_used = sum(a_mach[m,k] * x_v[m,k,t].X for k in K)
        reg_used   = max(0.0, hours_used - ot_val)
        rows2.append({
            'Furnace': m, 't': t,
            'ON (z)': z_val,
            'Reg. hrs used': round(reg_used, 4),
            'Overtime hrs':  round(ot_val, 4),
            'Total hrs used': round(hours_used, 4),
        })
df_furn = pd.DataFrame(rows2)
print("\nFurnace Table:")
print(df_furn.to_string(index=False))

# ===========================================================================
# 5. A.2 — HOLDING-COST SENSITIVITY
# ===========================================================================

print("\n" + "="*70)
print("  A.2 — HOLDING-COST SENSITIVITY  (alpha sweep, 20 points)")
print("="*70)

alphas   = np.linspace(0, 5, 20)
print(
    "Running sensitivity analysis on alpha (holding-cost multiplier):\n"
    f" Testing values: {', '.join(f'{a:.2f}' for a in alphas)} \n"
)
costs_A2 = []
invs_A2  = []

for alpha in alphas:
    m2, vs2 = build_and_solve(alpha=alpha, beta=1.0, output_flag=False)
    status_str = "Optimal" if m2 is not None and m2.Status == GRB.OPTIMAL else "Infeasible/Unbounded"
    if m2 is None:
        costs_A2.append(float('nan'))
        invs_A2.append(float('nan'))
    else:
        costs_A2.append(m2.ObjVal)
        total_inv = sum(vs2['inv'][k,t].X for k in K for t in T)
        invs_A2.append(total_inv)
    print(f"{status_str}  alpha={alpha:.3f}  cost={costs_A2[-1]:,.2f}  inv={invs_A2[-1]:.2f}")

kink_alpha = None # point where inventory drops to zero, indicating a shift from produce-ahead to just-in-time strategy
for a_val, inv_val in zip(alphas, invs_A2):
    if inv_val < 1e-3:
        kink_alpha = a_val
        break

if kink_alpha is not None:
    print(f"\n=> Produce-ahead strategy abandoned at alpha ≈ {kink_alpha:.4f}")
else:
    print("\n=> Inventory never reaches zero in the tested range.")

fig2, ax1 = plt.subplots(figsize=(9, 5))
color1, color2 = 'steelblue', 'firebrick'
ax2 = ax1.twinx()
l1, = ax1.plot(alphas, costs_A2, 'o-', color=color1, markersize=5, linewidth=1.8, label='Total Cost')
l2, = ax2.plot(alphas, invs_A2,  's--', color=color2, markersize=5, linewidth=1.8, label='Total Inventory')
ax1.set_xlabel(r'Holding-cost multiplier $\alpha$', fontsize=12)
ax1.set_ylabel('Total Cost ($)', color=color1, fontsize=12)
ax2.set_ylabel('Total End-of-Period Inventory (units)', color=color2, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax2.tick_params(axis='y', labelcolor=color2)
ax1.set_title(r'Holding-Cost Sensitivity: Total Cost and Inventory vs. $\alpha$', fontsize=13)
if kink_alpha is not None:
    ax1.axvline(x=kink_alpha, color='gray', linestyle=':', linewidth=1.5,
                label=f'Kink $\\alpha\\approx{kink_alpha:.2f}$')
ax1.legend(handles=[l1, l2] + ax1.get_lines()[1:], loc='center right', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
path_A2 = os.path.join(SCRIPT_DIR, 'fig_A2_holding_sensitivity.png')
fig2.savefig(path_A2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Figure saved: {path_A2}")

# ===========================================================================
# 6. A.3 — OVERTIME-COST SENSITIVITY
# ===========================================================================

print("\n" + "="*70)
print("  A.3 — OVERTIME-COST SENSITIVITY  (beta sweep, 15 points)")
print("="*70)

betas    = np.linspace(0.5, 4.0, 15)
print(
    "Running sensitivity analysis on beta (overtime-cost multiplier):\n"
    f" Testing values: {', '.join(f'{b:.2f}' for b in betas)} \n"
)
costs_A3 = []
ots_A3   = []

for beta in betas:
    m3, vs3 = build_and_solve(alpha=1.0, beta=beta, output_flag=False)
    status_str = "Optimal" if m3 is not None and m3.Status == GRB.OPTIMAL else "Infeasible/Unbounded"
    if m3 is None:
        costs_A3.append(float('nan'))
        ots_A3.append(float('nan'))
    else:
        costs_A3.append(m3.ObjVal)
        total_ot = sum(vs3['ot'][m,t].X for m in M for t in T)
        ots_A3.append(total_ot)
    print(f"{status_str}  beta={beta:.3f}  cost={costs_A3[-1]:,.2f}  overtime={ots_A3[-1]:.4f}")

breakeven_beta = None # point where overtime drops to zero, indicating that the overtime premium exceeds the marginal benefit of extra production
for b_val, ot_val in zip(betas, ots_A3):
    if ot_val < 1e-3:
        breakeven_beta = b_val
        break

if breakeven_beta is not None:
    print(f"\n=> Overtime abandoned at beta ≈ {breakeven_beta:.4f}")
    print(f"   Economic interpretation: at beta={breakeven_beta:.2f}, overtime premium "
          f"({breakeven_beta*15:.1f} $/hr) exceeds the marginal benefit of extra production.")
else:
    print("\n=> Overtime is used throughout the tested beta range.")

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax4 = ax3.twinx()
l3, = ax3.plot(betas, costs_A3, 'o-',  color='steelblue', markersize=5, linewidth=1.8, label='Total Cost')
l4, = ax4.plot(betas, ots_A3,   's--', color='darkorange', markersize=5, linewidth=1.8, label='Total Overtime hrs')
ax3.set_xlabel(r'Overtime-cost multiplier $\beta$', fontsize=12)
ax3.set_ylabel('Total Cost ($)', color='steelblue', fontsize=12)
ax4.set_ylabel('Total Overtime Hours', color='darkorange', fontsize=12)
ax3.tick_params(axis='y', labelcolor='steelblue')
ax4.tick_params(axis='y', labelcolor='darkorange')
ax3.set_title(r'Overtime-Cost Sensitivity: Total Cost and Overtime vs. $\beta$', fontsize=13)
if breakeven_beta is not None:
    ax3.axvline(x=breakeven_beta, color='gray', linestyle=':', linewidth=1.5,
                label=f'Breakeven $\\beta\\approx{breakeven_beta:.2f}$')
ax3.legend(handles=[l3, l4] + ax3.get_lines()[1:], loc='upper left', fontsize=10)
ax3.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
path_A3 = os.path.join(SCRIPT_DIR, 'fig_A3_overtime_sensitivity.png')
fig3.savefig(path_A3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"Figure saved: {path_A3}")

print("\n" + "="*70)
