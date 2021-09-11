#!/usr/bin/env python3
"""Calculates the radial distribution function.
"""

import math

from md import load_xyz

import numpy as np

def calculate_rdf(types, pos, r_min, r_max, dr, itype=None, jtype=None):
    """Calculates the RDF

    Does not consider periodic boundaries which is problematic when calculating
    the RDF of all atoms because the average system density is undefined and
    edge atoms will have a different local structure than bulk atoms.

    Args:
        types (list): Element type of each atom as a string
        pos (numpy.array): Each row is an atom, columns are x, y, z
        r_min (float): Lower limit of RDF we wish to calculate, in cm
        r_max (float): Upper limit of RDF we wish to calculate, in cm
        dr (float): Width of each RDF bin, in cm
        itype (str or None): Limit center atom (r=0) to this element
        itype (str or None): Limit pairs to this element

    Returns:
        tuple: Two numpy.array types.  First contains r values corresponding to
        each RDF bin, and second contains the normalized RDF at each bin.
    """
    # infer box dimensions
    xl = pos[:, 0].max()
    yl = pos[:, 1].max()
    zl = pos[:, 2].max()

    # initialize rdf
    rdf = np.zeros(int((r_max - r_min) // dr) + 2)
    rdf_x = np.linspace(r_min, r_max, rdf.shape[0])

    # count number of atoms in each bin
    for i in range(0, pos.shape[0] - 1):
        if itype and types[i] != itype:
            continue
        for j in range(i + 1, pos.shape[0]):
            if jtype and types[j] != jtype:
                continue
            dpos = pos[j, :] - pos[i, :]
            r = math.sqrt(np.dot(dpos, dpos))
            nbin = int((r - r_min) // dr)
            if nbin >= 0 and nbin < rdf.shape[0]:
                rdf[nbin] += 1

    # divide by volume of each bin
    for i in range(rdf.shape[0]):
        rdf[i] /= 4 * np.pi * rdf_x[i] * rdf_x[i] * dr

    # normalize rdf
    avg_dens = rdf.shape[0] / (xl * yl * zl)
    rdf /= avg_dens

    return rdf_x, rdf

def main():
    types, pos = load_xyz("input.xyz")

    # define min/max of our RDF
    r_min = 0.5e-8 # in cm
    r_max = 10.0e-8 # in cm
    dr = 0.25e-8 # in cm

    rdf_x, rdf = calculate_rdf(types, pos, r_min, r_max, dr)

    # print rdf
    print("{:12} {:12s}".format("r (cm)", "g(r)"))
    for i in range(rdf.shape[0]):
        print( "{:12.6e} {:12.6e}".format(rdf_x[i], rdf[i]))

if __name__ == "__main__":
    main()
