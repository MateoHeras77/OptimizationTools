from ortools.linear_solver import pywraplp

# Define the data
num_products = 5  # Number of products
num_suppliers = 3  # Number of suppliers
num_periods = 3  # Number of periods

# Storage space requirements for each product
storage_space = [10, 15, 20, 25, 30]

# Demand for each product in each period
demand = [
    [57, 90, 58, 92, 54],  # Period 1
    [72, 73, 95, 97, 88],  # Period 2
    [92, 89, 95, 53, 87]   # Period 3
]

# Unit purchasing cost for each product from each supplier
purchase_cost = [
    [209, 719, 503, 857, 530],  # Supplier 1
    [997, 362, 582, 731, 930],  # Supplier 2
    [578, 133, 750, 589, 467]   # Supplier 3
]

# Unit holding cost for each product
holding_cost = [15.00, 10.00, 15.00, 18.00, 16.00]

# Available storage space in each period
available_storage = [3500, 4200, 6000]

# Ordering cost for each supplier
ordering_cost = [5795, 1856, 4106]

# Create the solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Decision Variables
# X_ijt: Ordering quantity of product i from supplier j at time t
X = {}
for i in range(num_products):
    for j in range(num_suppliers):
        for t in range(num_periods):
            X[(i, j, t)] = solver.IntVar(0, solver.infinity(), f'X_{i}_{j}_{t}')

# Y_jt: Binary variable (1 if an order is placed from supplier j at time t, 0 otherwise)
Y = {}
for j in range(num_suppliers):
    for t in range(num_periods):
        Y[(j, t)] = solver.IntVar(0, 1, f'Y_{j}_{t}')

# I_it: Total inventory of product i at time t
I = {}
for i in range(num_products):
    for t in range(num_periods):
        I[(i, t)] = solver.IntVar(0, solver.infinity(), f'I_{i}_{t}')

# Objective Function: Minimize Total Cost = Ordering Cost + Purchasing Cost + Holding Cost
total_cost = 0

# Ordering Cost
for j in range(num_suppliers):
    for t in range(num_periods):
        total_cost += ordering_cost[j] * Y[(j, t)]

# Purchasing Cost
for i in range(num_products):
    for j in range(num_suppliers):
        for t in range(num_periods):
            total_cost += purchase_cost[j][i] * X[(i, j, t)]

# Holding Cost
for i in range(num_products):
    for t in range(num_periods):
        total_cost += holding_cost[i] * I[(i, t)]

solver.Minimize(total_cost)

# Constraints
# 1. Inventory Balance Constraint
for i in range(num_products):
    for t in range(num_periods):
        if t == 0:
            # Initial inventory is zero
            solver.Add(I[(i, t)] == sum(X[(i, j, t)] for j in range(num_suppliers)) - demand[t][i])
        else:
            solver.Add(I[(i, t)] == I[(i, t - 1)] + sum(X[(i, j, t)] for j in range(num_suppliers)) - demand[t][i])

# 2. X_ijt is positive only if an order is placed (Y_jt = 1)
M = 100000  # A large number
for i in range(num_products):
    for j in range(num_suppliers):
        for t in range(num_periods):
            solver.Add(X[(i, j, t)] <= M * Y[(j, t)])

# 3. Storage Space Constraints
for t in range(num_periods):
    solver.Add(sum(storage_space[i] * I[(i, t)] for i in range(num_products)) <= available_storage[t])


# Solve the problem
status = solver.Solve()

# Check if the solution is optimal
if status == pywraplp.Solver.OPTIMAL:
    print("Optimal Solution Found!")
    print(f"Minimum Total Cost: ${solver.Objective().Value():,.2f}")
else:
    print("No optimal solution found.")

if status == pywraplp.Solver.OPTIMAL:
    # Display Ordering Quantities (X_ijt)
    print("\nOrdering Quantities (X_ijt):")
    for t in range(num_periods):
        print(f"\nPeriod {t + 1}:")
        for i in range(num_products):
            for j in range(num_suppliers):
                if X[(i, j, t)].solution_value() > 0:
                    print(f"  Product {i + 1} from Supplier {j + 1}: {X[(i, j, t)].solution_value():.0f} units")

    # Display Inventory Levels (I_it)
    print("\nInventory Levels (I_it):")
    for t in range(num_periods):
        print(f"\nPeriod {t + 1}:")
        for i in range(num_products):
            print(f"  Product {i + 1}: {I[(i, t)].solution_value():.0f} units")

    # Display Ordering Decisions (Y_jt)
    print("\nOrdering Decisions (Y_jt):")
    for t in range(num_periods):
        print(f"\nPeriod {t + 1}:")
        for j in range(num_suppliers):
            if Y[(j, t)].solution_value() > 0:
                print(f"  Supplier {j + 1}: Order Placed (Y_{j + 1}_{t + 1} = 1)")

    # Cost Breakdown
    print("\nCost Breakdown:")
    ordering_cost_total = sum(ordering_cost[j] * Y[(j, t)].solution_value() for j in range(num_suppliers) for t in range(num_periods))
    purchasing_cost_total = sum(purchase_cost[j][i] * X[(i, j, t)].solution_value() for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods))
    holding_cost_total = sum(holding_cost[i] * I[(i, t)].solution_value() for i in range(num_products) for t in range(num_periods))

    print(f"  Total Ordering Cost: ${ordering_cost_total:,.2f}")
    print(f"  Total Purchasing Cost: ${purchasing_cost_total:,.2f}")
    print(f"  Total Holding Cost: ${holding_cost_total:,.2f}")