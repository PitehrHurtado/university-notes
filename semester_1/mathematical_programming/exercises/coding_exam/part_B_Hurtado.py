"""
part_B_Hurtado.py

Name: Pitehr Hurtado-Cayo
=================
Part B of the Coding Exam — Minimum Lot Size + Setup Cost extension.
"""

import os
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# Set script directory for saving figures and ouptuts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 1. SETS & PARAMETERS  (identical to Part A)
# ===========================================================================
K = ['A', 'B', 'C']
M = ['F1', 'F2']
J = ['DC1', 'DC2']
T = [1, 2, 3, 4]

D = {
    ('DC1','A',1):28, ('DC1','A',2):27, ('DC1','A',3):33, ('DC1','A',4):30,
    ('DC1','B',1):17, ('DC1','B',2):20, ('DC1','B',3):26, ('DC1','B',4):22,
    ('DC1','C',1):11, ('DC1','C',2):17, ('DC1','C',3):14, ('DC1','C',4):17,
    ('DC2','A',1):27, ('DC2','A',2):30, ('DC2','A',3):28, ('DC2','A',4):33,
    ('DC2','B',1):19, ('DC2','B',2):19, ('DC2','B',3):26, ('DC2','B',4):24,
    ('DC2','C',1):13, ('DC2','C',2):19, ('DC2','C',3):17, ('DC2','C',4):18,
}

c0_m  = {'F1': 8,  'F2': 10}
c0_k  = {'A': 0,  'B': 1,  'C': 2}
CP    = {(m, k, t): c0_m[m] + c0_k[k] for m in M for k in K for t in T}
CT    = {(j, k, t): (1 if j == 'DC1' else 2) for j in J for k in K for t in T}
CH    = {'A': 1.0, 'B': 1.2, 'C': 0.8}
CB    = {'A': 4,   'B': 5,   'C': 3}
a_mach = {(m, k): v for m in M for k, v in zip(K, [0.5, 0.7, 0.4])}
H_reg  = {(m, t): 50  for m in M for t in T}
H_ov   = {(m, t): 20  for m in M for t in T}
CF     = {(m, t): 200 for m in M for t in T}
cov    = {(m, t): 15  for m in M for t in T}
I0     = {'A': 0, 'B': 0, 'C': 0}
S_cap  = {'A': 200, 'B': 200, 'C': 200}
rho    = {'A': 0.60, 'B': 0.60, 'C': 0.60}
epsilon = 0.25

# Part B specific
L_min  = {'A': 25, 'B': 20, 'C': 15}
SC     = {(m, k): 50 for m in M for k in K}

# artigicial parameter used hardcoded
_M_big_k = {'A': 140, 'B': 100, 'C': 175}
M_big  = {(m, k, t): _M_big_k[k] for m in M for k in K for t in T}


# ===========================================================================
# 2. SHARED CONSTRAINT BUILDER (adds constraints (2)-(10))
# ===========================================================================

def _add_base_constraints(mdl, x, y, inv, bkd, ot, z):
    for k in K:
        for t in T:
            inv_prev = I0[k] if t == 1 else inv[k, t-1]
            mdl.addConstr(
                inv[k,t] == inv_prev
                           + quicksum(x[m,k,t] for m in M)
                           - quicksum(y[j,k,t] for j in J),
                name=f'inv_bal_{k}_{t}'
            )
    for k in K:
        for t in T:
            bkd_prev = 0 if t == 1 else bkd[k, t-1]
            mdl.addConstr(
                quicksum(y[j,k,t] for j in J) + bkd[k,t]
                == quicksum(D[j,k,t] for j in J) + bkd_prev,
                name=f'backorder_bal_{k}_{t}'
            )
    for k in K:
        mdl.addConstr(bkd[k, 4] == 0, name=f'backorder_final_{k}')
    for k in K:
        for t in T:
            mdl.addConstr(
                quicksum(y[j,k,t] for j in J)
                >= rho[k] * quicksum(D[j,k,t] for j in J),
                name=f'service_{k}_{t}'
            )
    for m in M:
        for t in T:
            mdl.addConstr(
                quicksum(a_mach[m,k] * x[m,k,t] for k in K)
                <= H_reg[m,t] * z[m,t] + ot[m,t],
                name=f'hours_used_{m}_{t}'
            )
    for m in M:
        for t in T:
            mdl.addConstr(
                ot[m,t] <= H_ov[m,t] * z[m,t],
                name=f'overtime_lim_{m}_{t}'    
            )
    for k in K:
        for t in T:
            if t >= 2:
                X_kt = quicksum(x[m,k,t]  for m in M)
                X_kt_1 = quicksum(x[m,k,t-1] for m in M)
                mdl.addConstr(X_kt <= (1 + epsilon) * X_kt_1, name=f'smooth_up_{k}_{t}')
                mdl.addConstr(X_kt >= (1 - epsilon) * X_kt_1, name=f'smooth_down_{k}_{t}')



# ===========================================================================
# 3. BUILD PART A (base model)
# ===========================================================================

def build_part_A():
    mdl = gp.Model("part_A_base")
    mdl.setParam("OutputFlag", 0)

    x   = mdl.addVars(M, K, T, vtype=GRB.CONTINUOUS, lb=0, name='x')
    y   = mdl.addVars(J, K, T, vtype=GRB.CONTINUOUS, lb=0, name='y')
    inv = mdl.addVars(K, T, vtype=GRB.CONTINUOUS,    lb=0, ub=200, name='inv')
    bkd = mdl.addVars(K, T, vtype=GRB.CONTINUOUS,    lb=0, name='bkd')
    ot  = mdl.addVars(M, T, vtype=GRB.CONTINUOUS,    lb=0, name='ot')
    z   = mdl.addVars(M, T, vtype=GRB.BINARY,              name='z')

    mdl.setObjective(
        quicksum(CP[m,k,t] * x[m,k,t]  for m in M for k in K for t in T) +
        quicksum(CT[j,k,t] * y[j,k,t]  for j in J for k in K for t in T) +
        quicksum(CF[m,t]   * z[m,t]    for m in M for t in T) +
        quicksum(cov[m,t]  * ot[m,t]   for m in M for t in T) +
        quicksum(CH[k]     * inv[k,t]  for k in K for t in T) +
        quicksum(CB[k]     * bkd[k,t]  for k in K for t in T),
        GRB.MINIMIZE
    )
    _add_base_constraints(mdl, x, y, inv, bkd, ot, z)
    mdl.optimize()
    return mdl, x


# ===========================================================================
# 4. BUILD PART B (extended model)
# ===========================================================================

def build_part_B():
    mdl = gp.Model("part_B_extended")
    mdl.setParam("OutputFlag", 0)

    x     = mdl.addVars(M, K, T, vtype=GRB.CONTINUOUS, lb=0, name='x')
    y     = mdl.addVars(J, K, T, vtype=GRB.CONTINUOUS, lb=0, name='y')
    inv   = mdl.addVars(K, T, vtype=GRB.CONTINUOUS,    lb=0, name='inventory')
    bkd   = mdl.addVars(K, T, vtype=GRB.CONTINUOUS,    lb=0, name='backorder')
    ot    = mdl.addVars(M, T, vtype=GRB.CONTINUOUS,    lb=0, name='overtime')
    z     = mdl.addVars(M, T, vtype=GRB.BINARY,              name='furnace_used')

    # delta[m,k,t] = 1 if we pay setup cost for producing product k in furnace m at time t, 0 otherwise
    delta = mdl.addVars(M, K, T, vtype=GRB.BINARY,           name='delta')

    # Objective: base costs + setup costs
    mdl.setObjective(
        quicksum(CP[m,k,t] * x[m,k,t]     for m in M for k in K for t in T) +
        quicksum(CT[j,k,t] * y[j,k,t]     for j in J for k in K for t in T) +
        quicksum(CF[m,t]   * z[m,t]        for m in M for t in T) +
        quicksum(cov[m,t]  * ot[m,t]       for m in M for t in T) +
        quicksum(CH[k]     * inv[k,t]      for k in K for t in T) +
        quicksum(CB[k]     * bkd[k,t]      for k in K for t in T) +
        quicksum(SC[m,k]   * delta[m,k,t]  for m in M for k in K for t in T),
        GRB.MINIMIZE
    )

    # Base constraints (2)-(10)
    _add_base_constraints(mdl, x, y, inv, bkd, ot, z)

    # New constraints: Big-M and minimum-lot
    for m in M:
        for k in K:
            for t in T:
                # Big-M (force x=0 when delta=0)
                mdl.addConstr(
                    x[m,k,t] <= M_big[m,k,t] * delta[m,k,t],
                    name=f'bigM_{m}_{k}_{t}'
                )
                # Minimum-lot (force x >= L_k when delta=1)
                mdl.addConstr(
                    x[m,k,t] >= L_min[k] * delta[m,k,t],
                    name=f'minlot_{m}_{k}_{t}'
                )

    mdl.optimize()
    return mdl, x, delta


# ===========================================================================
# 5. SOLVE, COMPARE, REPORT
# ===========================================================================

print("\n" + "="*70)
print("  B.2 — PART B: MINIMUM LOT SIZE + SETUP COST")
print("="*70)

print("\nSolving Part A base model...")
mdl_A, x_A = build_part_A()
print(f"Part A optimal cost  : {mdl_A.ObjVal:,.4f}")

print("\nSolving Part B extended model...")
mdl_B, x_B, delta_B = build_part_B()
print(f"Part B optimal cost    : {mdl_B.ObjVal:,.4f}")
print(f"Cost Delta (FO_B-FO_A) : {mdl_B.ObjVal - mdl_A.ObjVal:.4f}")

# ii) Count and list setups
setups = [(m, k, t) for m in M for k in K for t in T if delta_B[m,k,t].X > 0.5]
print(f"\nNumber of (m,k,t) setups paid: {len(setups)}")
print("Setup list:")
for m, k, t in setups:
    print(f"  delta[{m},{k},{t}]=1  x={x_B[m,k,t].X:.4f}")

# iii) Side-by-side comparison table X^(A) vs X^(B)
rows = []
for m in M:
    for k in K:
        for t in T:
            xA_val = x_A[m,k,t].X
            xB_val = x_B[m,k,t].X
            changed = '*' if abs(xA_val - xB_val) > 1e-3 else ''
            rows.append({
                'Furnace': m, 'Product': k, 't': t,
                'X^(A)':   round(xA_val, 4),
                'X^(B)':   round(xB_val, 4),
                'Delta':   round(xB_val - xA_val, 4),
                'Changed': changed,
            })
df_cmp = pd.DataFrame(rows)
print("\nSide-by-side comparison  X^(A) vs X^(B)   (* = changed):")
print(df_cmp.to_string(index=False))

# Aggregate total production per (k,t) in both models
print("\nAggregate X_{k,t} = sum_m x_{m,k,t}:")
rows3 = []
for k in K:
    for t in T:
        xA_sum = sum(x_A[m,k,t].X for m in M)
        xB_sum = sum(x_B[m,k,t].X for m in M)
        rows3.append({
            'Product': k, 't': t,
            'X^(A)_total': round(xA_sum, 4),
            'X^(B)_total': round(xB_sum, 4),
            'Diff': round(xB_sum - xA_sum, 4),
        })
df_agg = pd.DataFrame(rows3)
print(df_agg.to_string(index=False))

print("\n" + "="*70)
