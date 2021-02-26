# This module is part of the python standard library
import time

# These modules are part of other existing libraries
import numpy as np
import matplotlib.pyplot as plt

# This is my own script (it is an interface to the pybullet simulator)
import ae353_cmg

# I often go back and forth between making changes to my scripts and to
# the notebook in which they are used. One "gotcha" is that notebooks only
# import modules or scripts ONCE. Subsequent imports don't do anything, and
# in particular won't reflect any changes I've made to my scripts. To make
# sure that I'm working with the latest version of my code, I use this bit
# of magic, which forces the notebook to "reload" my script:



robot = ae353_cmg.RobotSimulator(damping=0., dt=0.001, display=True)

robot.snapshot('test.png')

class RobotController:
    def __init__(self, dt=0.001, q_2_des=0.):
        self.dt = dt
        
        # desired gimbal angle
        self.q_2_des = q_2_des
        
        # state feedback gains
        self.K = np.array([[5., 1.]])
        
        # equilibrium point
        self.q_2_e = self.q_2_des
        self.v_2_e = 0.
        self.tau_2_e = 0.
    
    def run(self, q_1, v_1, q_2, v_2, q_3, v_3):
        # state
        x = np.array([[q_2 - self.q_2_e], [v_2 - self.v_2_e]])
        
        # input
        u = - self.K @ x
        
        # actuator commands
        tau_2 = u[0, 0] + self.tau_2_e
        tau_3 = 0.
        return tau_2, tau_3

controller = RobotController(dt=robot.dt, q_2_des=-1.)

# Choose initial angles
q_1_0 =  1.25  # <-- initial platform angle
q_2_0 =  -0.10  # <-- initial gimbal angle
q_3_0 =  0.30  # <-- initial rotor angle (does it matter?)

# Choose initial angular velocities
v_1_0 =  0.05  # <-- initial platform angular velocity
v_2_0 = -0.05  # <-- initial gimbal angular velocity
v_3_0 = (100 * 2 * np.pi / 60) # <-- initial rotor angular velocity (100 rpm)

# Apply initial conditions
#
#  The two arguments are 1D numpy arrays of length 3. The first
#  array has the initial values of each angle. The second array
#  has the initial values of each angular velocity.
#
robot.set_state(np.array([q_1_0, q_2_0, q_3_0]), np.array([v_1_0, v_2_0, v_3_0]))


# IMPORTANT
#
#  The following code will override any initial conditions we have already
#  chosen. So if you want to use some other set of initial conditions, you
#  must comment out (or delete, or replace) the following lines:
#
# # Restore the simulation to its initial state
# robot.reset(rotor_rpm=100.)

# Choose how long we want to run the simulation, and
# compute the corresponding number of time steps
run_time = 10.
num_steps = int(run_time/robot.dt)

# Create a dictionary in which to store results
data = {
    't': np.empty(num_steps, dtype=float),
    'q_1': np.empty(num_steps, dtype=float),
    'v_1': np.empty(num_steps, dtype=float),
    'q_2': np.empty(num_steps, dtype=float),
    'v_2': np.empty(num_steps, dtype=float),
    'q_3': np.empty(num_steps, dtype=float),
    'v_3': np.empty(num_steps, dtype=float),
    'tau_2': np.empty(num_steps, dtype=float),
    'tau_3': np.empty(num_steps, dtype=float),
}

# Run the simulation loop
start_time = time.time()
for step in range(num_steps):
    # Get the current time
    t = robot.dt * step
    
    # Get the sensor measurements
    q_1, v_1, q_2, v_2, q_3, v_3 = robot.get_sensor_measurements()
    
    # Choose the actuator command (by running the controller)
    tau_2, tau_3 = controller.run(q_1, v_1, q_2, v_2, q_3, v_3)
    
    # Log the data from this time step
    data['t'][step] = t
    data['q_1'][step] = q_1
    data['v_1'][step] = v_1
    data['q_2'][step] = q_2
    data['v_2'][step] = v_2
    data['q_3'][step] = q_3
    data['v_3'][step] = v_3
    data['tau_2'][step] = tau_2
    data['tau_3'][step] = tau_3
    
    # Send the actuator commands to robot and go forward one time
    # step (this is where the actual simulation happens)
    robot.set_actuator_commands(tau_2, tau_3)
    robot.step(t=(start_time + (robot.dt * (step + 1))))


fig = plt.figure(figsize=(8, 6))
plt.plot(data['t'], data['q_2'], label='gimbal angle (rad)', linewidth=4)
plt.plot(data['t'], np.ones_like(data['t']) * controller.q_2_des, '--', label='desired gimbal angle (rad)', linewidth=4)
plt.plot(data['t'], data['v_2'], ':', label='gimbal angular velocity (rad/s)', linewidth=3)
plt.legend(fontsize=18)
plt.grid()
plt.tick_params(labelsize=16)
plt.xlabel('time (s)', fontsize=16)

plt.show()
