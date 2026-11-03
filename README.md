# 3D Wind Tunnel Flow Simulation using Physics-Informed Neural Networks (PINNs)

An advanced 3D aerodynamic flow simulation engine built around **DeepXDE**. This project trains a Fully Connected Neural Network (FNN) to solve the steady-state incompressible 3D Navier-Stokes equations inside a virtual wind tunnel containing an arbitrary geometry imported from a legacy Gmsh file.

## 🚀 Key Features

* **Legacy Mesh Parser:** Extract raw 3D spatial node coordinates directly from standard legacy Gmsh `.msh` (v2.2) files.
* **Constructive Geometry Masks:** Automatically calculates an interior solid bounding box void, preventing the PINN from unphysically evaluating fluid points inside the solid vehicle object.
* **Dual-Stage Optimization Handshake:**
  * **Stage 1 (Adam):** Employs a custom-built **Softmax Adaptive Weighting** callback engine that dynamically adjusts the multi-objective loss parameters on-the-fly, forcing the model to heavily prioritize strict no-slip conditions on the complex car body surface.
  * **Stage 2 (L-BFGS):** Freezes the optimal adaptive weight landscape to allow high-precision quasi-Newton fine-tuning without triggering numerical line-search failures.
* **High-Speed Flow Tracer & Animator:** Evaluates the fluid network state once across a user-specified 2D center-plane grid. It uses vectorized `SciPy` interpolators to simulate interactive, leak-free particle aerodynamic streamline paths, exporting directly to high-frame-rate `.gif` files.

---

## 🛠️ Pipeline Architecture

1. **Geometry Definition:** Parses a custom `assembly.msh` file, pads the limits to create an automated external wind tunnel bounding box, and constraints the domain using a custom logic class overriding DeepXDE's sampling routine.
2. **Physical Constraints:** Employs `PointSetBC` to clamp velocities ($u=0, v=0, w=0$) directly onto the discrete imported nodes alongside standard continuous analytical Dirichlet conditions for the tunnel boundary limits.
3. **Loss Mapping Blueprint:**
   $$\mathcal{L} = \omega_{pde}\mathcal{L}_{NS} + \omega_{in}\mathcal{L}_{Inlet} + \omega_{out}\mathcal{L}_{Outlet} + \omega_{wall}\mathcal{L}_{Walls} + \omega_{obj}\mathcal{L}_{NoSlip}$$

---

## 💻 Technical Setup

### Prerequisites

Ensure you have a backend engine configured for DeepXDE (TensorFlow, PyTorch, or JAX).

```bash
pip install deepxde numpy scipy matplotlib pillow
```

### File Hierarchy & Usage

1. Place your target layout mesh named as `assembly.msh` directly inside the root workspace folder.
2. Run your primary python script containing your custom training architecture loop.
3. Once calculations finish, check your working directory for the exported file: `airflow_simulation_car.gif`.

---

## 📊 Performance & Optimization Insights

* **Why we isolate the Softmax Callback from L-BFGS:** L-BFGS builds an internal approximation of the Hessian matrix over past positional updates. Modifying structural loss parameters midway shifts the underlying optimization landscape, resulting in immediate learning rate contraction or "Line search failed" crashes. Freezing the landscape after the Adam phase yields robust, steady convergence.
* **Leaking Buffers Solved:** The included animation script initializes graph primitive `Quiver` objects strictly once outside the rendering update sequence. Frame intervals update positions via memory-safe `.set_offsets()` and vector properties through `.set_UVC()`, circumventing costly runtime memory leaks.
