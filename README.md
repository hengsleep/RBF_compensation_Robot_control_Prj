# Adaptive Control of Robotic Manipulators with RBF and σ-Modification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB/Simulink](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

A Lyapunov-based adaptive control framework for trajectory tracking of uncertain robotic manipulators, integrating RBF neural network approximation and $\sigma$-modification adaptive laws for robust uncertainty compensation and parameter boundedness.

---

## Overview

This repository implements a nonlinear adaptive control framework for trajectory tracking of uncertain robotic manipulators.

The proposed architecture integrates computed torque control (CTC) with online RBF neural network approximation to estimate lumped system uncertainties, while a Lyapunov-based adaptive law with $\sigma$-modification guarantees closed-loop stability and prevents adaptive parameter drift.

### Key Features

* **Dual-Loop Architecture:** Outer-loop trajectory tracking separated from inner-loop dynamic compensation.
* **Lyapunov-Based Stability Analysis:** 
  Closed-loop stability analysis with Uniform Ultimate Boundedness (UUB) guarantee for the tracking error system.
* **σ-Modification Adaptive Law:** 
  Adaptive parameter update mechanism designed to prevent weight drift under persistent approximation errors.
* **RBF Neural Approximation:** 
  Online estimation of lumped nonlinear uncertainties and external disturbances.
* **Multi-System Verification:** 
  Validated on both 1-DOF nonlinear servo dynamics and 2-DOF MIMO robotic manipulators.

---

## System Architecture & Simulation Results

The simulation framework evaluates single-joint dynamics and 2-DOF robotic systems under various control configurations, demonstrating the robustness and accurate disturbance estimation of the proposed adaptive RBF strategy.

### 1. Single-Joint System Performance (Sim 1)

Comparison between nominal Computed Torque Control (CTC) without adaptive compensation and the proposed complete adaptive architecture with RBF disturbance approximation.

|                     Baseline (Mode$S=1$: Nominal CTC)                     |          Proposed (Mode$S=3, S_1=2, m=19$: CTC + RBF + Adaptive $\sigma$)          |
| :--------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| ![Sim1 Baseline](<python_result_pdf/readme_figures/Sim1_S=1%20Tracking.png>) | ![Sim1 Proposed](<python_result_pdf/readme_figures/Sim1_S=3_S1=2_m=19%20Tracking.png>) |
|       *Severe state divergence due to uncompensated model mismatch.*       |    *Precise trajectory tracking and asymptotic convergence via RBF compensation.*    |

---

### 2. Multi-Axis 2-DOF Robotic System Verification (Sim 2)

Comprehensive verification on a multi-input multi-output (MIMO) 2-DOF robotic manipulator under injected nonlinear disturbances and 20% model parameter mismatch.

#### State Tracking Profiles

| Position Tracking ($q_1$, $q_2$) | Speed Tracking ($\dot{q}_1$, $\dot{q}_2$) |
| :--------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| ![2DOF Position Tracking](python_result_pdf/readme_figures/Sim2_S=3_S1=2_m=5%20Position%20Tracking.png) | ![2DOF Speed Tracking](python_result_pdf/readme_figures/Sim2_S=3_S1=2_m=5%20Speed%20Tracking.png) |
| *Accurate angular position tracking for joint 1 and joint 2 under 20% model uncertainty.* | *Rapid transient response and velocity error convergence under nonlinear disturbances.* |

#### Disturbance Estimation

| Unknown Disturbance Approximation |
| :---: |
| ![Disturbance Estimation](python_result_pdf/readme_figures/Sim2_S=3_S1=2_m=5%20Disturbance%20Estimation.png) |
| *Real-time neural network approximation accurately matching external disturbance dynamics.* |

---

<details>
<summary><b>🔍 View Simulation Configurations & Parameter Descriptions</b></summary>

| Parameter                        | Description                  | Options / Values                                                                           |
| :------------------------------- | :--------------------------- | :----------------------------------------------------------------------------------------- |
| **Control Mode ($S$)**   | Primary control architecture | $S=1$ (Nominal CTC)$S=2$ (Ideal Feedforward)$S=3$ (Proposed CTC + RBF + Disturbance) |
| **Adaptive Law ($S_1$)** | Gain adjustment mechanism    | $S_1=1$ (Standard Adaptive)$S_1=2$ (With $\sigma$-modification)                      |
| **RBF Nodes ($m$)**      | Hidden layer size for RBF NN | $m = 5, 19, 50$                                                                          |

> **Note:** Full comparative run figures across all $m$ nodes and $S_1$ modes are archived in [`python_result_pdf/`](./python_result_pdf/).

</details>

---

## Control Architecture and Stability Analysis

The control design follows a Lyapunov-based adaptive control framework. 
The unknown nonlinear dynamics and external disturbances are treated as lumped uncertainties and approximated using an RBF neural network.

The rigid robotic manipulator dynamics are governed by the Euler-Lagrange equation:

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau + d
$$

The joint control torque $\tau$ is designed as:

$$
\tau = M_0(q)v + C_0(q,\dot{q})\dot{q} + G_0(q) - \hat{f}(x)
$$

where the virtual control acceleration $v$ guarantees second-order tracking dynamics ($v = \ddot{q}_d - K_v \dot{e} - K_p e$), and $\hat{f}(x)$ represents the lumped uncertainty approximated by the RBF network:

$$
\hat{f}(x) = \hat{W}^T h(x)
$$

To ensure robustness, the neural weights are updated online via the $\sigma$-modified adaptive law:

$$
\dot{\hat{W}} = \gamma h(X)X^T P B - \sigma \gamma \|X\| \hat{W}
$$

---

## Directory & Module Structure

| Path                            | File Name                     | Description                                                                    |
| :------------------------------ | :---------------------------- | :----------------------------------------------------------------------------- |
| **`/src/dynamics/`**    | `two_link_manipulator.m`    | Dynamic model ($M_0, C_0, G_0$) for a 2-DOF planar manipulator.              |
|                                 | `nonlinear_servo.slx`       | Simulink model for 1-DOF servo system with non-smooth friction.                |
| **`/src/controllers/`** | `computed_torque_control.m` | Outer-loop tracking controller calculating virtual acceleration commands.      |
|                                 | `rbf_compensator.m`         | Gaussian RBF network initialization and forward propagation for$\hat{f}(x)$. |
| **`/src/adaptation/`**  | `weight_update_law.m`       | Numerical solver for the$\sigma$-modified adaptive update law.               |
| **`/simulations/`**     | `run_servo_sim.m`           | Main evaluation script for the 1-DOF system under modes$S=1, 2, 3$.          |
|                                 | `run_two_link_sim.m`        | Main simulation script executing 2-DOF robust validation under uncertainties.  |

---

## Project Structure

```text
26 EXHIBITION/
├── LATEX/                                 # LaTeX source files for academic publication
│   └── main latex/
│       ├── sections/                      # Modular LaTeX chapters (Theory, Proofs, Sims)
│       │   ├── Controller Design.tex
│       │   ├── Problem Formulation.tex
│       │   ├── Simulation.tex
│       │   └── Stability Analysis.tex
│       ├── refs.bib                       # BibTeX reference database
│       └── bare_jrnl.tex                  # Main IEEE journal LaTeX entry point
│
├── matlab source code/                    # MATLAB/Simulink implementation
│   ├── simulation_1sim.mdl                # 1-DOF single-joint Simulink plant model
│   ├── simulation_1ctrl.m                 # 1-DOF RBF adaptive controller
│   ├── simulation_2sim.mdl                # 2-DOF planar manipulator Simulink model
│   ├── simulation_2ctrl.m                 # 2-DOF MIMO adaptive controller
│   └── simulation_2plot.m                 # Result plotting script for MATLAB
│
├── python/                                # Pure Python simulation framework
│   ├── sim1/                              # 1-DOF single-joint simulation suite
│   │   ├── controller.py                  # RBF neural network controller & update law
│   │   ├── plant.py                       # Single-joint dynamic equations
│   │   └── run_all.py                     # One-click execution for Sim1
│   ├── sim2/                              # 2-DOF planar manipulator simulation suite
│   │   ├── controller.py                  # MIMO adaptive RBF controller
│   │   ├── plant.py                       # 2-DOF Euler-Lagrange manipulator dynamics
│   │   └── run_all.py                     # One-click execution for Sim2
│   └── sim_7DOF_franka/                   # High-dimensional 7-DOF Franka Emika robot sim
│       ├── controller.py                  # 7-DOF joint space adaptive controller
│       └── run_all.py                     # Execution entry for 7-DOF manipulator
│
├── python_result_pdf/                     # Exported figures & README assets
│   ├── readme_figures/                    # 300 DPI PNG assets embedded in this README
│   ├── export_readme_figures.py           # Script to auto-convert key PDFs to PNGs
│   └── Sim1_*.pdf / Sim2_*.pdf            # Complete vector PDF results for all test modes
│
├── RBF robotic control.pdf                # Compiled full paper/report
└── README.md                              # Repository documentation
```

---

## Quick Start

### Prerequisites

* MATLAB R2021b or newer
* Simulink & Control System Toolbox
* Python 3.8+ (optional, for result visualization scripts)

### Running Simulations

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Adaptive-RBF-Robotic-Control.git](https://github.com/your-username/Adaptive-RBF-Robotic-Control.git)
   cd Adaptive-RBF-Robotic-Control
   ```
