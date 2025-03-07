import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp as pl

st.set_page_config(layout="wide", page_title="Inventory Optimization Tool")

def run_optimization(num_products, num_suppliers, num_periods, storage_space, demand, purchase_cost,
                    holding_cost, available_storage, ordering_cost):
    """Run the optimization model with the given parameters"""
    # Create the problem
    prob = pl.LpProblem("InventoryOptimization", pl.LpMinimize)
    
    # Decision Variables
    # X_ijt: Ordering quantity of product i from supplier j at time t
    X = {}
    for i in range(num_products):
        for j in range(num_suppliers):
            for t in range(num_periods):
                X[(i, j, t)] = pl.LpVariable(f'X_{i}_{j}_{t}', lowBound=0, cat='Integer')

    # Y_jt: Binary variable (1 if an order is placed from supplier j at time t, 0 otherwise)
    Y = {}
    for j in range(num_suppliers):
        for t in range(num_periods):
            Y[(j, t)] = pl.LpVariable(f'Y_{j}_{t}', cat='Binary')

    # I_it: Total inventory of product i at time t
    I = {}
    for i in range(num_products):
        for t in range(num_periods):
            I[(i, t)] = pl.LpVariable(f'I_{i}_{t}', lowBound=0, cat='Integer')

    # Objective Function: Minimize Total Cost = Ordering Cost + Purchasing Cost + Holding Cost
    ordering_cost_expr = pl.lpSum([ordering_cost[j] * Y[(j, t)] for j in range(num_suppliers) for t in range(num_periods)])
    purchasing_cost_expr = pl.lpSum([purchase_cost[j][i] * X[(i, j, t)] for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods)])
    holding_cost_expr = pl.lpSum([holding_cost[i] * I[(i, t)] for i in range(num_products) for t in range(num_periods)])
    
    prob += ordering_cost_expr + purchasing_cost_expr + holding_cost_expr, "Total Cost"

    # Constraints
    # 1. Inventory Balance Constraint
    for i in range(num_products):
        for t in range(num_periods):
            if t == 0:
                # Initial inventory is zero
                prob += I[(i, t)] == pl.lpSum([X[(i, j, t)] for j in range(num_suppliers)]) - demand[t][i], f"Inventory_Balance_{i}_{t}"
            else:
                prob += I[(i, t)] == I[(i, t - 1)] + pl.lpSum([X[(i, j, t)] for j in range(num_suppliers)]) - demand[t][i], f"Inventory_Balance_{i}_{t}"

    # 2. X_ijt is positive only if an order is placed (Y_jt = 1)
    M = 100000  # A large number
    for i in range(num_products):
        for j in range(num_suppliers):
            for t in range(num_periods):
                prob += X[(i, j, t)] <= M * Y[(j, t)], f"Order_Placement_{i}_{j}_{t}"

    # 3. Storage Space Constraints
    for t in range(num_periods):
        prob += pl.lpSum([storage_space[i] * I[(i, t)] for i in range(num_products)]) <= available_storage[t], f"Storage_Capacity_{t}"

    # Solve the problem
    prob.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=60))
    
    # Check if the solution is optimal
    if pl.LpStatus[prob.status] == 'Optimal':
        results = {
            "status": "optimal",
            "total_cost": pl.value(prob.objective),
            "ordering_quantities": {},
            "inventory_levels": {},
            "ordering_decisions": {},
            "cost_breakdown": {}
        }
        
        # Ordering Quantities (X_ijt)
        for t in range(num_periods):
            for i in range(num_products):
                for j in range(num_suppliers):
                    if pl.value(X[(i, j, t)]) > 0:
                        key = f"P{i+1}_S{j+1}_T{t+1}"
                        results["ordering_quantities"][key] = pl.value(X[(i, j, t)])

        # Inventory Levels (I_it)
        for t in range(num_periods):
            for i in range(num_products):
                key = f"P{i+1}_T{t+1}"
                results["inventory_levels"][key] = pl.value(I[(i, t)])

        # Ordering Decisions (Y_jt)
        for t in range(num_periods):
            for j in range(num_suppliers):
                key = f"S{j+1}_T{t+1}"
                results["ordering_decisions"][key] = pl.value(Y[(j, t)])

        # Cost Breakdown
        ordering_cost_total = sum(ordering_cost[j] * pl.value(Y[(j, t)]) 
                                  for j in range(num_suppliers) for t in range(num_periods))
        purchasing_cost_total = sum(purchase_cost[j][i] * pl.value(X[(i, j, t)]) 
                                    for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods))
        holding_cost_total = sum(holding_cost[i] * pl.value(I[(i, t)]) 
                                 for i in range(num_products) for t in range(num_periods))

        results["cost_breakdown"]["ordering_cost"] = ordering_cost_total
        results["cost_breakdown"]["purchasing_cost"] = purchasing_cost_total
        results["cost_breakdown"]["holding_cost"] = holding_cost_total
        
        # Additional data for visualization and analysis
        ordering_df = []
        for t in range(num_periods):
            for i in range(num_products):
                for j in range(num_suppliers):
                    if pl.value(X[(i, j, t)]) > 0:
                        ordering_df.append({
                            "Period": t+1,
                            "Product": i+1,
                            "Supplier": j+1,
                            "Quantity": pl.value(X[(i, j, t)])
                        })
        results["ordering_df"] = pd.DataFrame(ordering_df)
        
        inventory_df = []
        for t in range(num_periods):
            for i in range(num_products):
                inventory_df.append({
                    "Period": t+1,
                    "Product": i+1,
                    "Inventory": pl.value(I[(i, t)])
                })
        results["inventory_df"] = pd.DataFrame(inventory_df)
        
        return results, None
    else:
        return None, f"No optimal solution found. Status: {pl.LpStatus[prob.status]}"

def run_target_cost_optimization(num_products, num_suppliers, num_periods, storage_space, demand, purchase_cost,
                              holding_cost, available_storage, ordering_cost, target_cost):
    """Run optimization with a target cost constraint"""
    # Create the problem
    prob = pl.LpProblem("TargetCostOptimization", pl.LpMaximize)
    
    # Decision Variables
    # X_ijt: Ordering quantity of product i from supplier j at time t
    X = {}
    for i in range(num_products):
        for j in range(num_suppliers):
            for t in range(num_periods):
                X[(i, j, t)] = pl.LpVariable(f'X_{i}_{j}_{t}', lowBound=0, cat='Integer')

    # Y_jt: Binary variable (1 if an order is placed from supplier j at time t, 0 otherwise)
    Y = {}
    for j in range(num_suppliers):
        for t in range(num_periods):
            Y[(j, t)] = pl.LpVariable(f'Y_{j}_{t}', cat='Binary')

    # I_it: Total inventory of product i at time t
    I = {}
    for i in range(num_products):
        for t in range(num_periods):
            I[(i, t)] = pl.LpVariable(f'I_{i}_{t}', lowBound=0, cat='Integer')

    # Calculate total cost
    total_cost = pl.lpSum([ordering_cost[j] * Y[(j, t)] for j in range(num_suppliers) for t in range(num_periods)]) + \
                 pl.lpSum([purchase_cost[j][i] * X[(i, j, t)] for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods)]) + \
                 pl.lpSum([holding_cost[i] * I[(i, t)] for i in range(num_products) for t in range(num_periods)])
    
    # Add target cost constraint
    prob += total_cost <= target_cost, "Target_Cost_Constraint"
    
    # Objective: Maximize total units ordered (to find a solution that meets demand while staying under cost)
    prob += pl.lpSum([X[(i, j, t)] for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods)]), "Total Units"

    # Constraints
    # 1. Inventory Balance Constraint
    for i in range(num_products):
        for t in range(num_periods):
            if t == 0:
                # Initial inventory is zero
                prob += I[(i, t)] == pl.lpSum([X[(i, j, t)] for j in range(num_suppliers)]) - demand[t][i], f"Inventory_Balance_{i}_{t}"
            else:
                prob += I[(i, t)] == I[(i, t - 1)] + pl.lpSum([X[(i, j, t)] for j in range(num_suppliers)]) - demand[t][i], f"Inventory_Balance_{i}_{t}"

    # 2. X_ijt is positive only if an order is placed (Y_jt = 1)
    M = 100000  # A large number
    for i in range(num_products):
        for j in range(num_suppliers):
            for t in range(num_periods):
                prob += X[(i, j, t)] <= M * Y[(j, t)], f"Order_Placement_{i}_{j}_{t}"

    # 3. Storage Space Constraints
    for t in range(num_periods):
        prob += pl.lpSum([storage_space[i] * I[(i, t)] for i in range(num_products)]) <= available_storage[t], f"Storage_Capacity_{t}"

    # Solve the problem
    prob.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=60))
    
    # Check if the solution is optimal
    if pl.LpStatus[prob.status] == 'Optimal':
        # Calculate the actual cost of the solution
        actual_cost = pl.value(total_cost)
        
        results = {
            "status": "optimal",
            "total_cost": actual_cost,
            "ordering_quantities": {},
            "inventory_levels": {},
            "ordering_decisions": {},
            "cost_breakdown": {}
        }
        
        # Ordering Quantities (X_ijt)
        for t in range(num_periods):
            for i in range(num_products):
                for j in range(num_suppliers):
                    if pl.value(X[(i, j, t)]) > 0:
                        key = f"P{i+1}_S{j+1}_T{t+1}"
                        results["ordering_quantities"][key] = pl.value(X[(i, j, t)])

        # Inventory Levels (I_it)
        for t in range(num_periods):
            for i in range(num_products):
                key = f"P{i+1}_T{t+1}"
                results["inventory_levels"][key] = pl.value(I[(i, t)])

        # Ordering Decisions (Y_jt)
        for t in range(num_periods):
            for j in range(num_suppliers):
                key = f"S{j+1}_T{t+1}"
                results["ordering_decisions"][key] = pl.value(Y[(j, t)])

        # Cost Breakdown
        ordering_cost_total = sum(ordering_cost[j] * pl.value(Y[(j, t)]) 
                                  for j in range(num_suppliers) for t in range(num_periods))
        purchasing_cost_total = sum(purchase_cost[j][i] * pl.value(X[(i, j, t)]) 
                                    for i in range(num_products) for j in range(num_suppliers) for t in range(num_periods))
        holding_cost_total = sum(holding_cost[i] * pl.value(I[(i, t)]) 
                                 for i in range(num_products) for t in range(num_periods))

        results["cost_breakdown"]["ordering_cost"] = ordering_cost_total
        results["cost_breakdown"]["purchasing_cost"] = purchasing_cost_total
        results["cost_breakdown"]["holding_cost"] = holding_cost_total
        
        # Additional data for visualization and analysis
        ordering_df = []
        for t in range(num_periods):
            for i in range(num_products):
                for j in range(num_suppliers):
                    if pl.value(X[(i, j, t)]) > 0:
                        ordering_df.append({
                            "Period": t+1,
                            "Product": i+1,
                            "Supplier": j+1,
                            "Quantity": pl.value(X[(i, j, t)])
                        })
        results["ordering_df"] = pd.DataFrame(ordering_df)
        
        inventory_df = []
        for t in range(num_periods):
            for i in range(num_products):
                inventory_df.append({
                    "Period": t+1,
                    "Product": i+1,
                    "Inventory": pl.value(I[(i, t)])
                })
        results["inventory_df"] = pd.DataFrame(inventory_df)
        
        return results, None
    else:
        return None, f"No feasible solution found for the target cost. Status: {pl.LpStatus[prob.status]}"

# App title
st.title("Inventory Optimization Tool")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a mode:", 
                           ["Custom Parameters", "Best Solution", "Target Cost Optimization"])

# Default values from the given code
default_num_products = 5
default_num_suppliers = 3
default_num_periods = 3

default_storage_space = [10, 15, 20, 25, 30]

default_demand = [
    [57, 90, 58, 92, 54],  # Period 1
    [72, 73, 95, 97, 88],  # Period 2
    [92, 89, 95, 53, 87]   # Period 3
]

default_purchase_cost = [
    [209, 719, 503, 857, 530],  # Supplier 1
    [997, 362, 582, 731, 930],  # Supplier 2
    [578, 133, 750, 589, 467]   # Supplier 3
]

default_holding_cost = [15.00, 10.00, 15.00, 18.00, 16.00]

default_available_storage = [3500, 4200, 6000]

default_ordering_cost = [5795, 1856, 4106]

# Function to display results
def display_results(results):
    st.header("Optimization Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"${results['total_cost']:,.2f}")
    with col2:
        st.metric("Ordering Cost", f"${results['cost_breakdown']['ordering_cost']:,.2f}")
    with col3:
        st.metric("Purchasing Cost", f"${results['cost_breakdown']['purchasing_cost']:,.2f}")
    
    # Cost breakdown pie chart
    cost_data = {
        'Category': ['Ordering Cost', 'Purchasing Cost', 'Holding Cost'],
        'Cost': [
            results['cost_breakdown']['ordering_cost'],
            results['cost_breakdown']['purchasing_cost'],
            results['cost_breakdown']['holding_cost']
        ]
    }
    cost_df = pd.DataFrame(cost_data)
    
    fig_pie = px.pie(cost_df, values='Cost', names='Category', title='Cost Breakdown')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Ordering quantities visualization
    if 'ordering_df' in results and not results['ordering_df'].empty:
        st.subheader("Ordering Quantities")
        pivot_ordering = results['ordering_df'].pivot_table(
            index=['Period', 'Product'], 
            columns='Supplier', 
            values='Quantity', 
            fill_value=0
        )
        
        st.dataframe(pivot_ordering)
        
        # Bar chart for ordering quantities
        fig_orders = px.bar(
            results['ordering_df'], 
            x='Period', 
            y='Quantity', 
            color='Supplier',
            barmode='group',
            facet_row='Product',
            title='Ordering Quantities by Period, Product, and Supplier'
        )
        st.plotly_chart(fig_orders, use_container_width=True)
    
    # Inventory levels visualization
    if 'inventory_df' in results and not results['inventory_df'].empty:
        st.subheader("Inventory Levels")
        pivot_inventory = results['inventory_df'].pivot_table(
            index='Period', 
            columns='Product', 
            values='Inventory', 
            fill_value=0
        )
        
        st.dataframe(pivot_inventory)
        
        # Line chart for inventory levels
        fig_inventory = px.line(
            results['inventory_df'], 
            x='Period', 
            y='Inventory', 
            color='Product',
            markers=True,
            title='Inventory Levels by Period and Product'
        )
        st.plotly_chart(fig_inventory, use_container_width=True)

# 1. Custom Parameters Mode
if app_mode == "Custom Parameters":
    st.header("Custom Parameters")
    
    with st.expander("Problem Dimensions", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=default_num_products)
        with col2:
            num_suppliers = st.number_input("Number of Suppliers", min_value=1, max_value=10, value=default_num_suppliers)
        with col3:
            num_periods = st.number_input("Number of Periods", min_value=1, max_value=10, value=default_num_periods)
    
    with st.expander("Product Parameters", expanded=False):
        # Storage Space
        st.subheader("Storage Space Requirements")
        storage_space = []
        cols = st.columns(num_products)
        for i in range(num_products):
            default_value = default_storage_space[i] if i < len(default_storage_space) else 10
            storage_space.append(cols[i].number_input(f"Product {i+1}", min_value=1, value=default_value, key=f"storage_{i}"))
        
        # Holding Cost
        st.subheader("Holding Cost per Period")
        holding_cost = []
        cols = st.columns(num_products)
        for i in range(num_products):
            default_value = default_holding_cost[i] if i < len(default_holding_cost) else 10.0
            holding_cost.append(cols[i].number_input(f"Product {i+1}", min_value=0.0, value=default_value, key=f"holding_{i}"))
    
    with st.expander("Period Parameters", expanded=False):
        # Available Storage
        st.subheader("Available Storage Space")
        available_storage = []
        cols = st.columns(num_periods)
        for t in range(num_periods):
            default_value = default_available_storage[t] if t < len(default_available_storage) else 5000
            available_storage.append(cols[t].number_input(f"Period {t+1}", min_value=100, value=default_value, key=f"storage_avail_{t}"))
    
    with st.expander("Supplier Parameters", expanded=False):
        # Ordering Cost
        st.subheader("Ordering Cost")
        ordering_cost = []
        cols = st.columns(num_suppliers)
        for j in range(num_suppliers):
            default_value = default_ordering_cost[j] if j < len(default_ordering_cost) else 2000
            ordering_cost.append(cols[j].number_input(f"Supplier {j+1}", min_value=0, value=default_value, key=f"ordering_{j}"))
    
    with st.expander("Demand Data", expanded=False):
        st.subheader("Product Demand by Period")
        demand = []
        
        for t in range(num_periods):
            st.write(f"Period {t+1}")
            period_demand = []
            cols = st.columns(num_products)
            for i in range(num_products):
                default_value = default_demand[t][i] if t < len(default_demand) and i < len(default_demand[t]) else 50
                period_demand.append(cols[i].number_input(f"Product {i+1}", min_value=0, value=default_value, key=f"demand_{t}_{i}"))
            demand.append(period_demand)
    
    with st.expander("Purchase Cost Data", expanded=False):
        st.subheader("Purchase Cost by Supplier and Product")
        purchase_cost = []
        
        for j in range(num_suppliers):
            st.write(f"Supplier {j+1}")
            supplier_cost = []
            cols = st.columns(num_products)
            for i in range(num_products):
                default_value = default_purchase_cost[j][i] if j < len(default_purchase_cost) and i < len(default_purchase_cost[j]) else 500
                supplier_cost.append(cols[i].number_input(f"Product {i+1}", min_value=0, value=default_value, key=f"cost_{j}_{i}"))
            purchase_cost.append(supplier_cost)
    
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Calculating optimal inventory policy..."):
            results, error = run_optimization(num_products, num_suppliers, num_periods, storage_space, demand, purchase_cost,
                                           holding_cost, available_storage, ordering_cost)
            
            if error:
                st.error(error)
            else:
                display_results(results)

# 2. Best Solution Mode
elif app_mode == "Best Solution":
    st.header("Best Solution Using Default Parameters")
    
    st.write("This mode will calculate the optimal inventory policy using the default parameters.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Products", default_num_products)
    with col2:
        st.metric("Suppliers", default_num_suppliers)
    with col3:
        st.metric("Periods", default_num_periods)
    
    if st.button("Find Best Solution", type="primary"):
        with st.spinner("Calculating optimal inventory policy..."):
            results, error = run_optimization(default_num_products, default_num_suppliers, default_num_periods, 
                                           default_storage_space, default_demand, default_purchase_cost,
                                           default_holding_cost, default_available_storage, default_ordering_cost)
            
            if error:
                st.error(error)
            else:
                display_results(results)

# 3. Target Cost Optimization
elif app_mode == "Target Cost Optimization":
    st.header("Target Cost Optimization")
    
    st.write("""
    This mode will try to find a solution that meets your target cost constraint.
    Enter a target cost and the system will attempt to find a feasible solution.
    """)
    
    # Get the minimum possible cost by running the optimization once
    with st.spinner("Calculating the minimum possible cost..."):
        min_results, error = run_optimization(default_num_products, default_num_suppliers, default_num_periods, 
                                           default_storage_space, default_demand, default_purchase_cost,
                                           default_holding_cost, default_available_storage, default_ordering_cost)
    
    if error:
        st.error(error)
    else:
        min_cost = min_results["total_cost"]
        
        # Allow the user to set a target cost
        target_cost = st.slider("Target Cost ($)", 
                                min_value=int(min_cost), 
                                max_value=int(min_cost * 1.5), 
                                value=int(min_cost * 1.1),
                                step=1000)
        
        st.write(f"Minimum possible cost: ${min_cost:,.2f}")
        st.write(f"Your target cost: ${target_cost:,.2f}")
        
        if st.button("Find Optimal Solution for Target Cost", type="primary"):
            with st.spinner("Calculating solution for target cost..."):
                results, error = run_target_cost_optimization(default_num_products, default_num_suppliers, default_num_periods, 
                                                           default_storage_space, default_demand, default_purchase_cost,
                                                           default_holding_cost, default_available_storage, default_ordering_cost,
                                                           target_cost)
                
                if error:
                    st.error(error)
                else:
                    display_results(results)
                    
                    st.subheader("Cost Savings Analysis")
                    savings = min_cost - results["total_cost"]
                    savings_percent = (savings / min_cost) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Cost Savings", f"${savings:,.2f}")
                    with col2:
                        st.metric("Savings Percentage", f"{savings_percent:.2f}%")

# Footer
st.markdown("---")
st.markdown("Inventory Optimization Tool - Supply Chain Management")