import ctypes
import os
import numpy as np
from mujoco_py import MjModel, MjSim, MjViewer

# Load the model from the XML file
model = MjModel("alice_laber.xml")

# Create a simulation
sim = MjSim(model)

# Get joint indices
joints = ["fr_leg1_pitch1", "fr_leg1_pitch2", "fl_leg2_pitch1", "fl_leg2_pitch2", "br_leg3_pitch1", "br_leg3_pitch2", "bl_leg4_pitch1", "bl_leg4_pitch2"]
joint_indices = [sim.model.joint_name2id(joint) for joint in joints]

# Simple gait controller
def gait_controller(phase):
    amplitude = 0.8
    frequency = 1.0
    offset = np.pi / 2.0

    joint_angles = np.zeros(8)
    joint_angles[0] = np.sin(phase * frequency) * amplitude
    joint_angles[1] = np.sin(phase * frequency + offset) * amplitude
    joint_angles[2] = np.sin(phase * frequency + np.pi) * amplitude
    joint_angles[3] = np.sin(phase * frequency + np.pi + offset) * amplitude
    joint_angles[4] = np.sin(phase * frequency + np.pi) * amplitude
    joint_angles[5] = np.sin(phase * frequency + np.pi + offset) * amplitude
    joint_angles[6] = np.sin(phase * frequency) * amplitude
    joint_angles[7] = np.sin(phase * frequency + offset) * amplitude

    return joint_angles

# Simulation loop
def simulate(sim, time_step=0.01, duration=10):
    phase = 0.0
    num_steps = int(duration / time_step)
    for _ in range(num_steps):
        joint_angles = gait_controller(phase)
        sim.data.ctrl[joint_indices] = joint_angles

        sim.step()
        viewer.render()

        phase += 0.01

# Launch the viewer and start the simulation
viewer = MjViewer(sim)
simulate(sim)
