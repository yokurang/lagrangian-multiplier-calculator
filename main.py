import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from sympy import Eq, diff, lambdify, solve, symbols
from sympy.parsing.sympy_parser import parse_expr


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


class LagrangianMultiplier:
    def __init__(self, objective_func, constraint_func):
        self.objective = objective_func
        self.constraint = constraint_func
        self.lambda_symbol = symbols("lambda")
        self.steps = []
        self.solution = None  # Store the solution here
        self.equations = self._setup_equations()

    def _setup_equations(self):
        # Step 1
        self.steps.append(
            f"Step 1: Collect the objective function: {self.objective.expr} "
            f"and the constraint function: {self.constraint.expr}"
        )

        # Step 2
        grad_f = self.objective.gradient()
        grad_g = self.constraint.gradient()
        self.steps.append(
            f"Step 2: Calculate the gradients for the objective function: {grad_f} "
            f"and the gradients for the constraint function: {grad_g}"
        )

        # Step 3
        equations = []
        for df, dg in zip(grad_f, grad_g):
            equations.append(Eq(df, self.lambda_symbol * dg))
        equations.append(Eq(self.constraint.expr, 0))
        self.steps.append(
            f"Step 3: Set up the following linear system of equations: {equations}"
        )

        return equations

    def solve_equations(self):
        variables = self.objective.variables + [self.lambda_symbol]
        self.solution = solve(self.equations, variables)

        # Step 4
        self.steps.append(
            f"Step 4: Solving the linear system of equations provides the solution: {self.solution}"
        )

        return self.solution

    def calculate_optimum_value(self):
        """
        Calculate the value of the objective function at the optimal solution.
        """
        # Check if the solution has been solved
        if not self.solution:
            raise ValueError(
                "No solution has been found yet. Please call solve_equations first."
            )

        # Check if the solution is a dictionary (single solution case)
        if isinstance(self.solution, dict):
            # Extract the variable names (e.g., x, y) and their corresponding values from the dictionary
            variable_values = {
                var: self.solution[var] for var in self.objective.variables
            }
        # Check if the solution is a list of dictionaries or tuples (multiple solutions case)
        elif isinstance(self.solution[0], dict):
            variable_values = {
                var: self.solution[0][var] for var in self.objective.variables
            }
        elif isinstance(self.solution[0], tuple):
            # Extract the values from the tuple and map them to the variables
            variable_values = {
                var: value
                for var, value in zip(self.objective.variables, self.solution[0])
            }
        else:
            raise ValueError("Unexpected solution format")

        # Directly substitute the solution values into the objective function
        optimum_value = self.objective.expr.subs(variable_values)

        # Step 5: Show the optimum value of the objective function
        self.steps.append(
            f"Step 5: Calculate the optimum value of the objective function at the solution point: {optimum_value}"
        )

        # Evaluate the optimum value numerically
        return optimum_value.evalf()

    def print_steps(self):
        for step in self.steps:
            print(step)

    def print_equations(self):
        for i, eq in enumerate(self.equations):
            print(f"Equation {i+1}: {eq}")

    def plot_problem(self):
        num_vars = len(self.objective.variables)
        if num_vars > 3:
            print("Cannot plot functions with more than 3 variables.")
            return

        if not os.path.exists("plots"):
            os.makedirs("plots")

        if num_vars == 2:
            self._plot_2d()
        elif num_vars == 3:
            self._plot_3d()

    def _plot_2d(self):
        x, y = self.objective.variables
        f = lambdify((x, y), self.objective.expr, "numpy")
        g = lambdify((x, y), self.constraint.expr, "numpy")

        x_range = np.linspace(-10, 10, 100)
        y_range = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, levels=20)
        fig.colorbar(contour)

        ax.contour(X, Y, g(X, Y), levels=[0], colors="r")

        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_title("Objective Function and Constraint")

        plt.savefig("plots/2d_plot.png")
        plt.close()

    def _plot_3d(self):
        x, y, z = self.objective.variables
        f = lambdify((x, y, z), self.objective.expr, "numpy")
        g = lambdify((x, y, z), self.constraint.expr, "numpy")

        x_range = np.linspace(-5, 5, 20)
        y_range = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x_range, y_range)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        Z = f(X, Y, 0)
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

        Z_constraint = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_constraint[i, j] = solve(g(X[i, j], Y[i, j], z), z)[0]
        ax.plot_surface(X, Y, Z_constraint, color="r", alpha=0.5)

        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_zlabel(str(z))
        ax.set_title("Objective Function and Constraint")
        plt.colorbar(surf)

        plt.savefig("plots/3d_plot.png")
        plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Solve constrained optimization problems using Lagrange Multipliers."
    )
    parser.add_argument(
        "-o", "--objective", required=True, help="Objective function to optimize"
    )
    parser.add_argument("-c", "--constraint", required=True, help="Constraint function")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed steps"
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="Generate and save plots"
    )
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

    # Calculate the optimum value of the objective function
    optimum_value = lagrange.calculate_optimum_value()
    print(
        f"\nOptimum value of the objective function at the solution point: {optimum_value}"
    )

    if args.verbose:
        print("\nDetailed steps:")
        lagrange.print_steps()

    if args.graph:
        print("\nGenerating plot...")
        lagrange.plot_problem()
        print(f"Plot saved in 'plots' directory.")


if __name__ == "__main__":
    main()
