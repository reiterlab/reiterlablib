#!/usr/bin/python
"""Unit tests for converting different measures of tumor size"""

__author__ = 'Johannes REITER'
__date__ = 'January 4, 2021'

import pytest

from rll.convert import *


@pytest.mark.parametrize('d_cm, volume',
                         [(0.0, 0.0), (0.1, 5.236e-4), (1, 0.5236)])
def test_sphere_volume(d_cm, volume):

    assert diameter_volume(d_cm) == pytest.approx(volume, rel=1e-3)


# @pytest.mark.skip(reason='not yet implemented')
@pytest.mark.parametrize('d_cm, cells_per_cm3, n_cells',
                         [(0.0, 1e9, 0), (1, 1e9, 5.23598e8), (1, 0, 0)])
def test_diameter_cells(d_cm, cells_per_cm3, n_cells):

    assert diameter_cells(d_cm, cells_per_cm3=cells_per_cm3) == pytest.approx(n_cells, rel=1e-3)


@pytest.mark.parametrize('n_cells, cells_per_cm3, d_cm',
                         [(0.0, 1e9, 0), (1e9, 1e9, 1.2407)])
def test_cells_diameter(n_cells, cells_per_cm3, d_cm):

    assert cells_diameter(n_cells, cells_per_cm3=cells_per_cm3) == pytest.approx(d_cm, rel=1e-3)
