import sympy as sp
import numpy as np
from sympy import symbols, diff, solve, lambdify, Eq
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

class Function:
    def __init__(self, expr_string):
        self.expr_string = expr_string
        self.expr = parse_expr(expr_string)
        self.variables = list(self.expr.free_symbols)
        self.variables.sort(key=lambda s: s.name)

    def gradient(self):
        return [diff(self.expr, var) for var in self.variables]
    
    def print_function(self):
        return self.expr_string

def get_user_input():
    print("Enter the objective function f(...):")
    obj_func = input("f(...) = ")
    
    print("Enter the constraint function g(...) = 0:")
    const_func = input("g(...) = ")
    
    return obj_func, const_func

def setup_problem(obj_func, const_func):
    f = Function(obj_func)
    g = Function(const_func)
    
    # Ensure both functions have the same variables
    if set(f.variables) != set(g.variables):
        raise ValueError("Objective and constraint functions must have the same variables.")
    
    return f, g

def compute_gradients(f, g):
    grad_f = f.gradient()
    grad_g = g.gradient()
    
    return grad_f, grad_g

class LagrangianMultiplier:
    def __init__(self, objective_func, constraint_func):
        self.objective = objective_func
        self.constraint = constraint_func
        self.lambda_symbol = symbols('lambda')
        self.steps = []
        self.equations = self._setup_equations()

    def _setup_equations(self):
        # Step 1
        self.steps.append(f"Step 1: Collect the objective function: {self.objective.expr} "
                          f"and the constraint function: {self.constraint.expr}")

        # Step 2
        grad_f = self.objective.gradient()
        grad_g = self.constraint.gradient()
        self.steps.append(f"Step 2: Calculate the gradients for the objective function: {grad_f} "
                          f"and the gradients for the constraint function: {grad_g}")

        # Step 3
        equations = []
        for df, dg in zip(grad_f, grad_g):
            equations.append(Eq(df, self.lambda_symbol * dg))
        equations.append(Eq(self.constraint.expr, 0))
        self.steps.append(f"Step 3: Set up the following linear system of equations: {equations}")

        return equations

    def get_equations(self):
        return self.equations

    def print_equations(self):
        for i, eq in enumerate(self.equations):
            print(f"Equation {i+1}: {eq}")

    def solve_equations(self):
        variables = self.objective.variables + [self.lambda_symbol]
        solution = solve(self.equations, variables)
        
        # Step 4
        self.steps.append(f"Step 4: Solving the linear system of equations provides the solution: {solution}")
        
        return solution

    def print_steps(self):
        for step in self.steps:
            print(step)

    def plot_problem(self):
        num_vars = len(self.objective.variables)
        if num_vars > 3:
            print("Cannot plot functions with more than 3 variables.")
            return
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        if num_vars == 2:
            self._plot_2d()
        elif num_vars == 3:
            self._plot_3d()

    def _plot_2d(self):
        x, y = self.objective.variables
        f = lambdify((x, y), self.objective.expr, 'numpy')
        g = lambdify((x, y), self.constraint.expr, 'numpy')

        x_range = np.linspace(-10, 10, 100)
        y_range = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, levels=20)
        fig.colorbar(contour)

        ax.contour(X, Y, g(X, Y), levels=[0], colors='r')

        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_title('Objective Function and Constraint')
        
        plt.savefig('plots/2d_plot.png')
        plt.close()

    def _plot_3d(self):
        x, y, z = self.objective.variables
        f = lambdify((x, y, z), self.objective.expr, 'numpy')
        g = lambdify((x, y, z), self.constraint.expr, 'numpy')

        x_range = np.linspace(-5, 5, 20)
        y_range = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x_range, y_range)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Z = f(X, Y, 0)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        Z_constraint = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_constraint[i,j] = solve(g(X[i,j], Y[i,j], z), z)[0]
        ax.plot_surface(X, Y, Z_constraint, color='r', alpha=0.5)

        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_zlabel(str(z))
        ax.set_title('Objective Function and Constraint')
        plt.colorbar(surf)
        
        plt.savefig('plots/3d_plot.png')
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Solve constrained optimization problems using Lagrange Multipliers.")
    parser.add_argument("-o", "--objective", required=True, help="Objective function to optimize")
    parser.add_argument("-c", "--constraint", required=True, help="Constraint function")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed steps")
    parser.add_argument("-g", "--graph", action="store_true", help="Generate and save plots")
    return parser.parse_args()
        
def main():
    args = parse_arguments()

    obj_func = Function(args.objective)
    const_func = Function(args.constraint)
    lagrange = LagrangianMultiplier(obj_func, const_func)

    print("Objective function:")
    print(f"f(...) = {obj_func.print_function()}")
    
    print("\nConstraint function:")
    print(f"g(...) = {const_func.print_function()}")
    
    print("Lagrangian equations:")
    lagrange.print_equations()
    
    print("\nSolution:")
    solution = lagrange.solve_equations()
    print(solution)
    
    if args.verbose:
        print("\nDetailed steps:")
        lagrange.print_steps()
    
    if args.graph:
        print("\nGenerating plot...")
        lagrange.plot_problem()
        print(f"Plot saved in 'plots' directory.")

if __name__ == "__main__":
    main()
