# Lagrange Multiplier Solver

This Python script solves constrained optimization problems using the method of Lagrange Multipliers.

## Features

- Solves optimization problems with up to 3 variables
- Generates 2D or 3D plots of the objective function and constraint
- Provides detailed step-by-step solution process

## Requirements

- Python 3.x
- SymPy
- NumPy
- Matplotlib

## Usage

```
python main.py -o "objective_function" -c "constraint_function" [-v] [-g]
```

- `-o`: Objective function to optimize
- `-c`: Constraint function
- `-v`: (Optional) Verbose mode, prints detailed steps
- `-g`: (Optional) Generate and save plots

## Example

```
python main.py -o "x**2 + 4*y**2 - 2*x + 8*y" -c "x + 2*y - 7" -v -g
```

Refer to `examples.txt` for more usage examples.

## Output

- Prints the objective and constraint functions
- Displays Lagrangian equations
- Provides the solution (optimal values)
- Generates plots (if -g flag is used)

For more information, contact [Alan Matthew/alan.matthew.yk@gmail.com].
