"""
Problema de Transporte Balanceado
==================================
Minimizar el costo total de enviar unidades desde plantas de origen
hacia centros de distribución, respetando capacidades de oferta y
satisfaciendo exactamente la demanda de cada destino.

Formulación:
    min  sum_{i in I, j in J} c[i,j] * x[i,j]
    s.t. sum_{j in J} x[i,j]  <= s[i]   para todo i in I   (oferta)
         sum_{i in I} x[i,j]  == d[j]   para todo j in J   (demanda)
         x[i,j] >= 0                    para todo i, j      (no negatividad)

El problema está balanceado: sum(s) == sum(d), por eso la demanda
se impone como igualdad (toda la demanda debe cubrirse exactamente).
"""

import gurobipy as gb
from gurobipy import GRB, quicksum
import pandas as pd

# ===========================================================================
# 1. DATOS SINTÉTICOS
# ===========================================================================

# Conjuntos de índices
I = ["Planta_A", "Planta_B", "Planta_C"]           # orígenes (plantas)
J = ["CD_Norte", "CD_Centro", "CD_Sur", "CD_Este"]  # destinos (centros de distribución)

# Oferta disponible por planta [unidades]
s = {
    "Planta_A": 120,
    "Planta_B":  80,
    "Planta_C": 100,
}  # total oferta = 300

# Demanda requerida por centro de distribución [unidades]
d = {
    "CD_Norte" :  70,
    "CD_Centro":  90,
    "CD_Sur"   :  80,
    "CD_Este"  :  60,
}  # total demanda = 300  → problema balanceado

# Costo unitario de transporte c[i,j] [$/unidad]
c = {
    ("Planta_A", "CD_Norte"):   2,
    ("Planta_A", "CD_Centro"):  3,
    ("Planta_A", "CD_Sur"):     8,
    ("Planta_A", "CD_Este"):   10,
    ("Planta_B", "CD_Norte"):   7,
    ("Planta_B", "CD_Centro"):  5,
    ("Planta_B", "CD_Sur"):     4,
    ("Planta_B", "CD_Este"):    6,
    ("Planta_C", "CD_Norte"):  12,
    ("Planta_C", "CD_Centro"):  9,
    ("Planta_C", "CD_Sur"):     3,
    ("Planta_C", "CD_Este"):    2,
}

# ===========================================================================
# 2. MODELO
# ===========================================================================

m = gb.Model("transporte")
m.setParam("OutputFlag", 0)

# ===========================================================================
# 3. VARIABLES DE DECISIÓN
# ===========================================================================

# X[i,j]: unidades enviadas desde origen i hacia destino j
X = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name="x")

# ===========================================================================
# 4. FUNCIÓN OBJETIVO
# ===========================================================================

m.setObjective(
    quicksum(c[i, j] * X[i, j] for i in I for j in J),
    GRB.MINIMIZE,
)

# ===========================================================================
# 5. RESTRICCIONES
# ===========================================================================

# Restricciones de oferta: no se puede despachar más de lo disponible
oferta_constrs = m.addConstrs(
    (quicksum(X[i, j] for j in J) <= s[i] for i in I),
    name="oferta",
)

# Restricciones de demanda: cada destino debe recibir exactamente su demanda
demanda_constrs = m.addConstrs(
    (quicksum(X[i, j] for i in I) == d[j] for j in J),
    name="demanda",
)

# ===========================================================================
# 6. OPTIMIZACIÓN Y SOLUCIÓN
# ===========================================================================

m.optimize()

if m.Status != GRB.OPTIMAL:
    print(f"[ERROR] El modelo no encontró solución óptima. Status: {m.Status}")
    raise SystemExit(1)

print("=" * 60)
print("SOLUCIÓN ÓPTIMA — PROBLEMA DE TRANSPORTE")
print("=" * 60)
print(f"Costo total mínimo: ${m.ObjVal:,.2f}\n")

# Tabla de flujos (solo rutas activas)
flujos = [
    {"Origen": i, "Destino": j, "Unidades": X[i, j].X, "Costo unitario": c[i, j], "Costo total": c[i, j] * X[i, j].X}
    for i in I for j in J
    if X[i, j].X > 1e-6
]
df_flujos = pd.DataFrame(flujos)
print("Rutas activas:")
print(df_flujos.to_string(index=False))

# Matriz de envíos (origen × destino)
print("\nMatriz de envíos [unidades]:")
matriz = pd.DataFrame(
    [[X[i, j].X for j in J] for i in I],
    index=I,
    columns=J,
)
print(matriz.to_string())

# ===========================================================================
# 7. ANÁLISIS DE SENSIBILIDAD
# ===========================================================================

print("\n" + "=" * 60)
print("ANÁLISIS DE SENSIBILIDAD — RESTRICCIONES")
print("=" * 60)

# Sensibilidad de restricciones: precios sombra y rangos de RHS
filas_constrs = []
for i in I:
    constr = oferta_constrs[i]
    filas_constrs.append({
        "Restricción": f"oferta_{i}",
        "Tipo"       : "Oferta (≤)",
        "RHS"        : s[i],
        "Holgura"    : constr.Slack,
        "Precio sombra (Pi)": constr.Pi,
        "RHS mín"    : constr.SARHSLow,
        "RHS máx"    : constr.SARHSUp,
    })
for j in J:
    constr = demanda_constrs[j]
    filas_constrs.append({
        "Restricción": f"demanda_{j}",
        "Tipo"       : "Demanda (=)",
        "RHS"        : d[j],
        "Holgura"    : constr.Slack,
        "Precio sombra (Pi)": constr.Pi,
        "RHS mín"    : constr.SARHSLow,
        "RHS máx"    : constr.SARHSUp,
    })

df_sens_constrs = pd.DataFrame(filas_constrs)
print(df_sens_constrs.to_string(index=False))

print("""
Interpretación:
  Precio sombra (Pi): cuánto aumenta el costo óptimo por cada unidad adicional
    de demanda o reducción de oferta. Un Pi negativo en oferta indica que liberar
    capacidad en esa planta reduciría el costo.
  Holgura: capacidad de oferta no utilizada (solo para restricciones ≤).
  RHS mín / máx: rango de valores del RHS en que el precio sombra se mantiene
    constante (base óptima no cambia).""")

print("\n" + "=" * 60)
print("ANÁLISIS DE SENSIBILIDAD — VARIABLES (COSTOS REDUCIDOS)")
print("=" * 60)

# Sensibilidad de variables: costos reducidos y rangos del coeficiente de costo
filas_vars = []
for i in I:
    for j in J:
        filas_vars.append({
            "Ruta"              : f"{i} → {j}",
            "Flujo"            : X[i, j].X,
            "Costo actual"     : c[i, j],
            "Costo reducido"   : X[i, j].RC,
            "Costo mín"        : X[i, j].SAObjLow,
            "Costo máx"        : X[i, j].SAObjUp,
        })

df_sens_vars = pd.DataFrame(filas_vars)
print(df_sens_vars.to_string(index=False))

print("""
Interpretación:
  Costo reducido (RC): cuánto debería bajar el costo unitario de una ruta
    inactiva (flujo == 0) para que convenga usarla. Para rutas activas, RC == 0.
  Costo mín / máx: rango del costo unitario c[i,j] en que la solución base
    (qué rutas se usan) permanece óptima sin recalcular. Si el costo sale de
    ese rango, la ruta puede entrar o salir de la solución.""")
