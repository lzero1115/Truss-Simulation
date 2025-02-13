# Grid Truss Structure Simulation

This project simulates the deformation of a grid truss structure.

## Setup and Configuration

1. **Adjust Truss Settings:**
   - Open `simulation.py` and modify the values of `nx`, `ny`, and `nz` to define the grid dimensions of the truss.
   - You can also modify the position of a single force point in the grid by adjusting its coordinates in the script.

2. **Launch the Simulation:**
   - Run the script to start the simulation.

## Using the Polyscope Viewer

1. In the Polyscope viewer, you can configure the following parameters:
   - **Global Applied Force:** Specify the force applied to the truss structure by inputting the value.
   - **Self Weight Consideration:** Decide whether to account for the self-weight of the truss rods.

2. After configuring the settings, press the **"Compute Truss Deformation"** button to compute the deformation of the truss under the applied force.

## External force
<img src="images/external_force.png" alt="img1" width="600"/>   
## Self weight
<img src="images/self_weight.png" alt="img2" width="600"/>  
