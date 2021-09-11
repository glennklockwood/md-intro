#!/usr/bin/env python3
"""Demonstrates basic concepts intrinsic to molecular dynamics simulations.
"""

import math
import argparse

import numpy as np

ATOMIC_MASS = {
    "ar": 39.948 / 6.022e23,
}

LENNARD_JONES_PARAMS = {
    "ar": {
        "ar": {
            "epsilon": 120.0 * 1.38064852e-16, # in ergs
            "sigma": 3.5e-8, # in cm
        },
    },
}

def main():
    """Runs simulation loop.

    Initialize atom positions, velocities, accelerations, and masses, then run
    the simulation loop.
    """
    types, pos = load_xyz("input.xyz")
    vel = np.zeros(pos.shape)
    accel = np.zeros(pos.shape)
    mass = np.full(pos.shape[0], ATOMIC_MASS["ar"])

    # The main simulation loop, where each iteration is a simulation timestep
    for i in range(20000):
        velocity_verlet(pos, vel, accel, mass, types, 1.0e-15)
        if (i % 100) == 0:
            print_xyz(pos)

def velocity_verlet(pos, vel, accel, mass, types, dt):
    """Calculates new atomic positions using Velocity Verlet.

    Use the Velocity Verlet algorithm to calculate the new positions,
    velocities, and accelerations for each atom based on their masses
    and the time step, dt.
    """
    force = np.zeros(pos.shape)
    energy = np.zeros(mass.shape)

    # Velocity Verlet, part I
    for i in range(len(pos)):
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]
        vel[i, 2] += 0.5 * dt * accel[i, 2]
        pos[i, 0] += dt * vel[i, 0]
        pos[i, 1] += dt * vel[i, 1]
        pos[i, 2] += dt * vel[i, 2]

    # Calculate energies and forces on each atom in this new configuration.
    # Note that we loop over pairs only once and use Newton's third law to
    # update each atom per pair.
    for i in range(0, len(pos) - 1):
        for j in range(i+1, len(pos)):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r = math.sqrt(dx*dx + dy*dy + dz*dz)

            epsilon = LENNARD_JONES_PARAMS[types[i]][types[j]]["epsilon"]
            sigma = LENNARD_JONES_PARAMS[types[i]][types[j]]["sigma"]
            force_ij, energy_ij = calculate_force_energy_lj(r, epsilon, sigma)

            energy[i] += energy_ij
            energy[j] += energy_ij
            force[i, 0] -= force_ij * dx / r
            force[i, 1] -= force_ij * dy / r
            force[i, 2] -= force_ij * dz / r
            force[j, 0] += force_ij * dx / r
            force[j, 1] += force_ij * dy / r
            force[j, 2] += force_ij * dz / r

    # Velocity Verlet, part II
    for i in range(len(pos)):
        accel[i, 0] = force[i, 0] / mass[i]
        accel[i, 1] = force[i, 1] / mass[i]
        accel[i, 2] = force[i, 2] / mass[i]
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]
        vel[i, 2] += 0.5 * dt * accel[i, 2]

def calculate_force_energy_lj(r, epsilon, sigma):
    """Calculates energy and force between two atoms.

    Calculate the potential energy and force between two atoms specified by
    distance r.  Currently uses a simple Lennard Jones potential with
    parameters for argon and returns a tuple of two scalars, the force and
    potential energy.
    """
    energy_ij = 4.0 * epsilon * ((sigma / r)**6.0 - (sigma / r)**12.0)
    force_ij = 24.0 * epsilon * \
        (2.0 * sigma**12.0/r**13.0 - sigma**6.0/r**7.0)

    return force_ij, energy_ij

def print_xyz(pos):
    """Prints atom positions in XYZ format.

    Print out the current position of every atom in the standard XYZ format.
    This output can then be fed into a visualization program like VMD.
    """
    print("%d" % len(pos))
    print("")
    for i in range(len(pos)):
        print("  {:2s}  {:10.6f} {:10.6f} {:10.6f}".format(
            "Ar",
            pos[i, 0] * 1.0e8,
            pos[i, 1] * 1.0e8,
            pos[i, 2] * 1.0e8
        ))

def load_xyz(filename):
    """Loads an XYZ file.

    Args:
        filename (str): Path to file containing XYZ data

    Returns:
        tuple: list of atom types, numpy.array of x, y, z positions for each
            atom in xyz file
    """
    with open(filename, "r") as xyzfile:
        num_atoms = int(xyzfile.readline())
        xyzfile.readline()
        positions = np.zeros((num_atoms, 3))
        types = []
        for i in range(num_atoms):
            atom_type, x, y, z = xyzfile.readline().strip().split()
            types.append(atom_type.lower())
            positions[i, :] = [float(x), float(y), float(z)]

        return types, positions * 1.0e-8

if __name__ == '__main__':
    main()
