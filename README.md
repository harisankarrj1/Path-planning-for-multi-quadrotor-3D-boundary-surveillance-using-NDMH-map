# Path-planning-for-multi-quadrotor-3D-boundary-surveillance-using-NDMH-map
Demonstration of hyperchaotic systems enable the generation of mutually independent trajectories with slight variations in initial conditions, allowing multiple quadrotors to operate simultaneously along a single guiding path.




This repository contains the code and simulation files associated with the research paper:
"Leveraging Hyperchaotic Systems for Autonomous 3D Quadrotor Surveillance"

Paper Link:
https://arxiv.org/pdf/2410.05215

Contents:
---------
1. **Folder Structure**:
   - `Fig/`: This folder contains the Python scripts used for the simulations and figures in the paper.

2. **Description of Code Files**:
   - `drone_3d_trajectory_following.py`: Simulates a quadrotor drone following a 3D trajectory with predefined waypoints provided as a list.
   - `droneqdmtraj11d.py`: Uses the QDM map as a hyperchaotic attractor to generate iterative sequences for quadrotor simulation along the hyperchaotic trajectories.

   - `dronendmhmap.py`: Simulates a quadrotor following a 3D trajectory, leveraging waypoints assigned as a list.
   - `droneqdmtraj11d.py`: Implements the NDMH map as a hyperchaotic attractor to generate iterative sequences for quadrotor simulation along the hyperchaotic trajectories.
A simulation video demonstrating the quadrotor following a hyperchaotic trajectory, which was used to generate Figure 11 in the paper, is available on YouTube:

Watch the simulation video on YouTube
https://youtu.be/Qco9w5fg_0g
This video showcases the quadrotor's motion along the hyperchaotic trajectory generated using the NDMH map, as discussed in the paper.
3. **Simulation Library**:
   - The simulations utilize the **PyRobotics** library for robotic modeling and control.
   - To learn more about the library, visit the official documentation or community resources.
   - Other programs in the repository are part of the PyRobotics library.

4. **How to Use**:
   - Ensure the PyRobotics library and its dependencies are installed before running the scripts.
   - Navigate to the `Fig/` folder and run the desired Python script to reproduce the corresponding figure or simulation mentioned in the paper.

5. **Instructions**:
   - For detailed information on the theoretical background and methodology, refer to the research paper.
   - Match the script filenames with the figures and discussions in the paper for better understanding.

6. **Citation**:
   - If you use this repository or any part of the associated work, kindly cite the paper as follows:

     ```
     @misc{r2024pathplanningmultiquadrotor3d,
      title={Path planning for multi-quadrotor 3D boundary surveillance using non-autonomous discrete memristor hyperchaotic system}, 
      author={Harisankar R and Abhishek Kaushik and Sishu Shankar Muni},
      year={2024},
      eprint={2410.05215},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2410.05215}, 
     }
     ```

7. **Disclaimer**:
   - The code provided is for academic and research purposes only.
   - For questions or clarifications, refer to the contact details mentioned in the paper.

8. **License**:
   - The repository is licensed under the MIT License. Please see the `LICENSE` file for details.

