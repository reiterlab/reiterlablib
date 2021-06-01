#!/usr/bin/python
"""Converting different measures of tumor sizes and tumor growth"""

import math

__date__ = 'December 21, 2020'
__author__ = 'Johannes REITER'


def diameter_volume(d_cm):
    """
    Calculates the volume of a sphere from the given diameter
    :param d_cm: sphere diameter in centimeters
    :return: volume of sphere
    """
    return 4.0 / 3 * (d_cm / 2) ** 3 * math.pi


def diameter_cells(d_cm, cells_per_cm3=1e9):
    """
    Converts the diameter of a sphere to the number of cells
    :param d_cm: sphere diameter in centimeters
    :param cells_per_cm3: number of cells per cubic centimeter (default: 1 billion cells)
    :return: number of cells
    """
    return diameter_volume(d_cm) * cells_per_cm3


def cells_diameter(n_cells, cells_per_cm3=1e9):
    """
    Takes number of cells and returns approximate spherical diameter in cm for an assumed number of cells per cm3
    :param n_cells: number of cells per cubic centimeters
    :param cells_per_cm3: number of cells per cubic centimeter (default: 1 billion cells)
    :return: spherical diameter in centimeters
    """
    return (6 * n_cells/cells_per_cm3 / math.pi) ** (1.0/3)


def volume_diameter(volume):
    """
    Takes volume in cm3 and returns approximate spherical diameter in cm
    :param volume: tumor volume in cm3
    :return: spherical diameter in centimeters
    """
    return (6 * volume / math.pi) ** (1.0/3)


def convert_tvdt_grrate(tvdt):
    """
    Converts the tumor volume doubling time in days to an exponential growth rate per day
    :param tvdt: tumor volume doubling time in days
    :return: exponential growth rate per day
    """
    return math.log(2) / tvdt


def convert_grrate_tvdt(r):
    """
    Converts a daily growth rate to the tumor volume doubling time in days
    :param r: daily growth rate
    :return: tumor volume doubling time in days
    """
    return math.log(2) / r
