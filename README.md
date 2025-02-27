# Computational Mechanics Project

This project combines Finite Element Methods (FEM), Physics-Informed Neural Networks (PINNs), and heart modeling to solve various elasticity problems. 

## Project Structure

- `FEM/`: Implementation of Finite Element Methods for linear and hyperelastic problems in 2D and 3D
- `pinns/`: Physics-Informed Neural Networks implementations for elasticity problems
- `heart_model/`: Contains heart geometry and mesh files for cardiac simulations
- `data/`: Generated data from simulations and experiments
- `experiments/`: Scripts for running experiments and generating plots

## Running Experiments

The `experiments/` directory contains several Python scripts for different types of experiments:

### Linear Elasticity
```bash
# 2D Linear Elasticity
python experiments/linear_elasticity_2D_experiment.py

# 3D Linear Elasticity
python experiments/linear_elasticity_3D_experiment.py

# Heart Linear Elasticity
python experiments/linear_elasticity_heart_experiment.py
```

### Hyperelasticity
```bash
# 2D Hyperelasticity
python experiments/hyper_elasticity_2D_experiment.py

# 3D Hyperelasticity
python experiments/hyper_elasticity_3D_experiment.py

# Heart Hyperelasticity
python experiments/hyper_elasticity_heart_experiment.py
```

### Data Generation
To generate training data for the PINNs:
```bash
python experiments/create_fenics_linear_2D_data.py
python experiments/create_fenics_linear_3D_data.py
python experiments/create_fenics_hyper_2D_data.py
python experiments/create_fenics_hyper_3D_data.py
```

## Results
Experiment results are automatically saved in the following directories:
- `experiments/plots_linear_2D/`
- `experiments/plots_linear_3D/`
- `experiments/plots_hyper_2D/`
- `experiments/plots_hyper_3D/`
- `experiments/plots_heart_linear/`
- `experiments/plots_heart_hyper/`

## Dependencies
- FEniCS
- PyTorch
- matplotlib
- numpy
- cardiac_geometries