#!/usr/bin/python
"""Common plotting methods"""

import logging
import numpy as np
import matplotlib.pyplot as plt

__date__ = 'November 5, 2020'
__author__ = 'Johannes REITER'


# get logger
logger = logging.getLogger(__name__)


def plot_histogram(data, xlim, ylim=None, n_xticks=None, n_yticks=None, density=True, bin_weights=None,
                   n_bins=15, rwidth=0.9, xlabel=None, ylabel=None, title=None,
                   figsize=(3.6, 2.7), align='mid', highlight_patches=None,
                   bar_color='dimgrey', lbl_fontsize=12, tick_fontsize=11, output_fp=None):
    """
    Plot a histogram of the given data
    :param data: array-like data
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param n_xticks: number of tick marks on the x-axis
    :param n_yticks: number of tick marks on the y-axis
    :param density: if true return a probability density (default True)
    :param bin_weights: array-like weights to normalize each bin or None (default None)
    :param n_bins: number of bins
    :param rwidth: width of bins
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: plot title
    :param figsize: figure size given as a tuple
    :param align: bin alignment
    :param highlight_patches: list of bin indexes to highlight
    :param bar_color: color of bars
    :param lbl_fontsize: font size of axis labels
    :param tick_fontsize: font size of tick marks
    :param output_fp: path to pdf output file
    :return: tuple of list of bin values and list of bin borders
    """

    fig, ax = plt.subplots(figsize=figsize)

    if n_yticks is not None:
        if ylim is None:
            ylim = ax.get_ylim()
        yticks = np.linspace(ylim[0], ylim[1], n_yticks)
        ax.set_yticks(yticks)

    alpha = 1.0
    if density:
        weights = np.ones_like(data) / float(len(data))
    else:
        weights = None

    bins = np.linspace(int(xlim[0]), int(xlim[1]), n_bins + 1)

    if bin_weights is not None:
        # because all but the las (righthand-most) bin is half-open,
        # we need to manually exclude values exactly at the right edge
        counts, bins = np.histogram(data[data < xlim[1]], bins=bins)
        data = bins[:-1]
        weights = counts * bin_weights

    bin_values, bin_borders, patches = plt.hist(data, bins=bins, align=align, rwidth=rwidth,
                                                weights=weights, color=bar_color, alpha=alpha)

    logger.debug('Bin values and borders: '
                 + ', '.join(f'[{start:.2e},{end:.2e}]: {val:.1e}'
                             for start, end, val in zip(bin_borders[:-1], bin_borders[1:], bin_values)))

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    ax.set_xlim((bin_borders[0] - (xlim[1] - xlim[0]) * 0.01, bin_borders[-1] + (xlim[1] - xlim[0]) * 0.01))
    if n_xticks is not None:
        xticks = np.linspace(xlim[0], xlim[1], n_xticks)
        ax.set_xticks(xticks)

    if highlight_patches is not None:
        logger.info('Highlighting patches in histogram: '
                    + ', '.join(f'{i}: {bin_values[i]:.3e}' for i in highlight_patches))
        for i in highlight_patches:
            patches[i].set_facecolor('firebrick')

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=lbl_fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=lbl_fontsize)

    # change the fontsize of ticks labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    #     ax.text(xlim[0]+(xlim[1]-xlim[0])*0.6, ylim[1]*0.9, 'mean: {:.4g}'.format(np.mean(data)),
    #             backgroundcolor='white', ha='left', color=blue, size=11, transform=ax.transData)
    #     ax.text(xlim[0]+(xlim[1]-xlim[0])*0.6, ylim[1]*0.75, 'median: {:.4g}'.format(np.median(data)),
    #             backgroundcolor='white', ha='left', color=blue, size=11, transform=ax.transData)

    set_axis_style(ax, xlim, ylim, outward=3)

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)

    return bin_values, bin_borders


def plot_xy(xs, yss, xlim=None, ylim=None, legend=True, legend_loc='best', bbox_to_anchor=None, leg_ncol=1,
            xlog=False, ylog=False, n_xticks=None, sci_notation_axes=None, x_offset_text_pos=None,
            xlabel=None, ylabel=None, title=None, labels=None, colors=None, markers=None,
            linestyle='-', linewidth=1.3, alpha=1.0,
            figsize=(3.6, 2.7), output_fp=None):
    """
    Create xy line plot
    :param xs: list of x positions
    :param yss: list of list of y positions
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param legend: show legend (default True)
    :param legend_loc: legend locations best 0 (default), upper right 1, upper left 2, lower left 3, lower right 4,
                        right 5, center left 6, center right 7, lower center 8, upper center 9, center 10
    :param bbox_to_anchor: position where legend should be anchored;
                           if a 4-tuple is given, then it specifies the legend as  (x pos, y pos, width, height)
    :param leg_ncol: number of columns in legend
    :param xlog: have x axis in logarithmic scale
    :param ylog: have y axis in logarithmic scale
    :param n_xticks: number of x-axis ticks
    :param sci_notation_axes:
    :param x_offset_text_pos:
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: plot title
    :param labels: labels for the individual lines in the legend
    :param colors: list of colors for the individual lines
    :param markers: list of markers for the individual lines
    :param linestyle: -, :-, --
    :param linewidth: width of the lines
    :param alpha: transparency of lines between 0 and 1 (default 1)
    :param figsize: figure size given as a tuple
    :param output_fp: path to pdf output file
    :return: list of Line2D objects
    """

    fig, ax = plt.subplots(figsize=figsize)

    lines = []
    for i, ys in enumerate(yss):
        lines.append(ax.plot(xs, ys, linestyle=linestyle, lw=linewidth, alpha=alpha, clip_on=False,
                     fillstyle='none', color='dimgrey' if colors is None else colors[i],
                     marker=None if markers is None else markers[i],
                     label=None if labels is None else labels[i]))

    if xlim is not None:
        ax.set_xlim(xlim)
        if n_xticks is not None:
            if not xlog:
                xticks = np.linspace(xlim[0], xlim[1], n_xticks)
                ax.set_xticks(xticks, minor=True if n_xticks > 10 else False)
    else:
        xlim = ax.get_xlim()

    # change given axis to scientific notation
    if sci_notation_axes is not None:
        for a in sci_notation_axes:
            ax.ticklabel_format(axis=a, style="sci", scilimits=(0, 0))

    if x_offset_text_pos is not None:
        # need to update the plot to be able to get the offset text
        plt.draw()
        x_offset_text = ax.xaxis.get_offset_text().get_text()
        ax.xaxis.get_offset_text().set_visible(False)
        ax.text(x_offset_text_pos[0], x_offset_text_pos[1], x_offset_text, fontsize=12)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)

    set_axis_style(ax, xlim, ylim, outward=5)

    plt.tick_params(axis='both', which='major', labelsize=11)

    if legend and labels is not None:
        if bbox_to_anchor is None:
            leg = plt.legend(loc=legend_loc, facecolor='white', frameon=True, framealpha=1.0, fancybox=False,
                             ncol=leg_ncol, prop={'size': 10})
        else:
            leg = plt.legend(loc=legend_loc, facecolor='white', frameon=True, framealpha=1.0, fancybox=False,
                             bbox_to_anchor=bbox_to_anchor, ncol=leg_ncol, prop={'size': 10})
        leg.get_frame().set_facecolor('white')
        # set the alpha value of the legend: it will be translucent
        leg.get_frame().set_alpha(1)
        leg.get_frame().set_linewidth(0.0)

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)
        logger.info(f'Created xy line plot: {output_fp}')

    return lines


def set_axis_style(ax, xlim, ylim, outward=0):
    """
    Remove top and right axis of box around the plot
    Set end of x-axis and y-axis according to the given limits
    :param ax:
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param outward: amount of space to separate left and bottom axis
    :return:
    """

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

    for line in ['left', 'bottom']:
        ax.spines[line].set_position(('outward', outward))
