from ortools.linear_solver import pywraplp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Crear el solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Parámetros del problema
products = range(5)  # 5 productos
time_periods = range(3)  # 3 periodos
suppliers = range(3)  # 3 proveedores

D = np.array([[57, 72, 92],
              [90, 73, 89],
              [58, 95, 95],
              [92, 97, 53],
              [54, 88, 87]])  # Demandas

P = np.array([[209, 997, 578],
              [719, 362, 133],
              [503, 582, 750],
              [857, 731, 589],
              [530, 930, 467]])  # Costos de compra

H = [15, 10, 15, 18, 16]  # Costos de almacenamiento
O = [5795, 1856, 4106]  # Costos de orden
W = [3500, 4200, 6000]  # Espacio disponible en cada periodo
f = [10, 15, 20, 25, 30]  # Espacio requerido por unidad
M = 10000  # Valor grande

# Variables de decisión
X = {}  # Cantidad ordenada
Y = {}  # Orden (binario)
I = {}  # Inventario

for i in products:
    for j in suppliers:
        for t in time_periods:
            X[i, j, t] = solver.NumVar(0, solver.infinity(), f'X_{i}_{j}_{t}')

for j in suppliers:
    for t in time_periods:
        Y[j, t] = solver.BoolVar(f'Y_{j}_{t}')

for i in products:
    for t in time_periods:
        I[i, t] = solver.NumVar(0, solver.infinity(), f'I_{i}_{t}')

# Función objetivo: minimizar costos
total_cost = solver.Sum(O[j] * Y[j, t] for j in suppliers for t in time_periods) + \
             solver.Sum(P[i, j] * X[i, j, t] for i in products for j in suppliers for t in time_periods) + \
             solver.Sum(H[i] * I[i, t] for i in products for t in time_periods)
solver.Minimize(total_cost)

# Restricción de balance de inventario
for i in products:
    for t in time_periods:
        if t == 0:
            solver.Add(I[i, t] == solver.Sum(X[i, j, t] for j in suppliers) - D[i, t])
        else:
            solver.Add(I[i, t] == I[i, t-1] + solver.Sum(X[i, j, t] for j in suppliers) - D[i, t])

# Restricción de orden
for i in products:
    for j in suppliers:
        for t in time_periods:
            solver.Add(X[i, j, t] <= M * Y[j, t])

# Restricción de espacio de almacenamiento
for t in time_periods:
    solver.Add(solver.Sum(f[i] * I[i, t] for i in products) <= W[t])

# Resolver el problema
status = solver.Solve()

# Mostrar resultados
if status == pywraplp.Solver.OPTIMAL:
    print(f'Solución óptima encontrada con costo mínimo: ${solver.Objective().Value():,.2f}')
    orders = np.zeros((5, 3, 3))
    inventory = np.zeros((5, 3))
    
    results = []
    for i in products:
        for t in time_periods:
            inventory[i, t] = I[i, t].solution_value()
            results.append([f'Producto {i+1}', f'Periodo {t+1}', f'{inventory[i, t]:.2f} unidades'])
        for j in suppliers:
            for t in time_periods:
                orders[i, j, t] = X[i, j, t].solution_value()
                if orders[i, j, t] > 0:
                    results.append([f'Producto {i+1}', f'Proveedor {j+1}', f'Periodo {t+1}', f'{orders[i, j, t]:.2f} unidades'])
    
    df_results = pd.DataFrame(results, columns=['Producto', 'Atributo 1', 'Atributo 2', 'Cantidad'])
    print(df_results.to_string(index=False))
    
    # Graficar inventario
    plt.figure(figsize=(12, 6))
    for i in products:
        plt.plot(time_periods, inventory[i, :], marker='o', linestyle='-', label=f'Producto {i+1}')
    plt.xlabel('Periodo')
    plt.ylabel('Inventario')
    plt.title('Niveles de Inventario por Producto')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Graficar ordenes
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.3
    for j in suppliers:
        orders_by_supplier = [sum(orders[i, j, t] for i in products) for t in time_periods]
        ax.bar(np.array(time_periods) + j * bar_width, orders_by_supplier, bar_width, label=f'Proveedor {j+1}')
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Cantidad Ordenada')
    ax.set_title('Órdenes por Proveedor en Cada Periodo')
    ax.legend()
    ax.grid()
    plt.xticks(time_periods, [f'Periodo {t+1}' for t in time_periods])
    plt.show()
else:
    print('No se encontró solución óptima.')
