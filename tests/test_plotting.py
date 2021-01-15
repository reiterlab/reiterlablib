#!/usr/bin/python
"""Unit tests for plotting methods"""

__author__ = 'Johannes REITER'
__date__ = 'December 11, 2020'

import pytest
import numpy as np

from rll.plotting import plot_histogram, plot_xy, plot_barplot


@pytest.mark.parametrize('data, xlim, n_bins, density, bin_values, bin_borders',
                         [([0, 1, 1, 2, 2, 2], (0, 3), 3, False, [1, 2, 3], [0, 1, 2, 3]),
                          ([1, 1, 2, 2, 4], (0, 5), 2, True, [0.8, 0.2], [0, 2.5, 5]), ])
def test_plot_histogram(data, xlim, n_bins, density, bin_values, bin_borders):

    bin_vals, bin_bors = plot_histogram(data, xlim, n_bins=n_bins, density=density)

    np.testing.assert_array_almost_equal(bin_vals, bin_values)
    np.testing.assert_array_almost_equal(bin_bors, bin_borders)


def test_plot_barplot():

    xs = list(range(0, 5))
    ys = list(range(0, 5))

    bar_container = plot_barplot(xs, ys)

    assert len(bar_container) == len(xs)
    assert len(bar_container) == len(ys)
    bar_heights = [rect.get_height() for rect in bar_container]
    np.testing.assert_array_almost_equal(bar_heights, ys)
    # np.testing.assert_array_almost_equal(lines[0][0].get_data()[0], list(range(0, 5)))
    # np.testing.assert_array_almost_equal(lines[1][0].get_data()[1], [1]*5)


def test_plot_xy():

    xs = list(range(0, 5))
    yss = [list(range(0, 5)), [1]*5]

    lines = plot_xy(xs, yss)

    assert len(lines) == len(yss)
    np.testing.assert_array_almost_equal(lines[0][0].get_data()[0], list(range(0, 5)))
    np.testing.assert_array_almost_equal(lines[1][0].get_data()[1], [1]*5)
