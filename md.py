#!/usr/bin/env python3
"""Demonstrates basic concepts intrinsic to molecular dynamics simulations.
"""

import math

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
            print_xyz(types, pos)

def velocity_verlet(pos, vel, accel, mass, types, dt):
    """Calculates new atomic positions using Velocity Verlet.

    Use the Velocity Verlet algorithm to calculate the new positions,
    velocities, and accelerations for each atom based on their masses
    and the time step, dt.
    """
    force = np.zeros(pos.shape)
    energy = np.zeros(mass.shape)

    # Velocity Verlet, part I
    for i in range(pos.shape[0]):
        vel[i, :] += 0.5 * dt * accel[i, :]
        pos[i, :] += dt * vel[i, :]

    # Calculate energies and forces on each atom in this new configuration.
    # Note that we loop over pairs only once and use Newton's third law to
    # update each atom per pair.
    for i in range(0, pos.shape[0] - 1):
        for j in range(i+1, pos.shape[0]):
            dpos = pos[j, :] - pos[i, :]
            r = math.sqrt(np.dot(dpos, dpos))

            epsilon = LENNARD_JONES_PARAMS[types[i]][types[j]]["epsilon"]
            sigma = LENNARD_JONES_PARAMS[types[i]][types[j]]["sigma"]
            force_ij, energy_ij = calculate_force_energy_lj(r, epsilon, sigma)

            energy[i] += energy_ij
            energy[j] += energy_ij
            force[i, :] -= force_ij * dpos / r
            force[j, :] += force_ij * dpos / r

    # Velocity Verlet, part II
    for i in range(pos.shape[0]):
        accel[i, :] = force[i, :] / mass[i]
        vel[i, :] += 0.5 * dt * accel[i, :]

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

def print_xyz(types, pos):
    """Prints atom positions in XYZ format.

    Print out the current position of every atom in the standard XYZ format.
    This output can then be fed into a visualization program like VMD.
    """
    print(pos.shape[0])
    print("")
    for i in range(pos.shape[0]):
        print("  {:2s}  {:10.6f} {:10.6f} {:10.6f}".format(
            types[i].title(),
            *(pos[i, :] * 1.0e8)))

def load_xyz(filename):
    """Loads an XYZ file.

    Args:
        filename (str): Path to file containing XYZ data

    Returns:
        tuple: list of atom types, numpy.array of x, y, z positions for each
            atom in xyz file
    """
    with open(filename, "r") as xyzfile:
        while True:
            try:
                num_atoms = int(xyzfile.readline())
            except ValueError:
                break
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
