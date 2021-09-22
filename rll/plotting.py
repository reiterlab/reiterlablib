#!/usr/bin/python
"""Common plotting methods"""

import logging
import math
import numpy as np
import matplotlib.pyplot as plt

__date__ = 'November 5, 2020'
__author__ = 'Johannes REITER'


# get logger
logger = logging.getLogger(__name__)

# default font size of x-axis and y-axis labels
LABEL_FS = 12

# default font size of tick mark labels
TICK_FS = 11

COLOR_PAL = ['dimgrey', 'firebrick', 'royalblue']
MARKERS = ['x', '+', '2', '.']

FIG_SIZE = (3.0, 2.6)


def plot_histogram(data, xlim, ylim=None, n_xticks=None, n_yticks=None, xticklabels=None, density=True, xlog=False,
                   bin_weights=None, n_bins=15, bin_borders=None, rwidth=0.9, xlabel=None, ylabel=None,
                   title=None, axes_separation=3, figsize=FIG_SIZE, align='mid', highlight_patches=None, notes=None,
                   clip_on=False, bar_color='dimgrey', alpha=1.0, lbl_fontsize=LABEL_FS, tick_fontsize=TICK_FS,
                   ax=None, output_fp=None):
    """
    Plot a histogram of the given data
    :param data: array-like data (or array of array-like data)
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param n_xticks: number of tick marks on the x-axis
    :param n_yticks: number of tick marks on the y-axis
    :param xticklabels: x-axis tick labels
    :param density: if true return a probability density (default True)
    :param xlog: have x axis in logarithmic scale
    :param bin_weights: array-like weights to normalize each bin or None (default None)
    :param n_bins: number of bins
    :param bin_borders: instead of the number of bins an array-like list of bin borders can be given (default None)
    :param rwidth: width of bins
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: plot title
    :param axes_separation: style parameter for the separation of x and y axes
    :param figsize: figure size given as a tuple
    :param align: bin alignment (default: 'mid')
    :param highlight_patches: list of bin indexes to highlight
    :param notes: dictionary of dictionaries where the key is a tuple of x and y position of the text label and the
                  values specifies various other properties
    :param clip_on: should lines be visible outside of the axes (default: False)
    :param bar_color: color of bars (or array)
    :param alpha: opacity of bars
    :param lbl_fontsize: font size of axis labels
    :param tick_fontsize: font size of tick marks
    :param ax: axes object of an already created figure
    :param output_fp: path to pdf output file
    :return: tuple of list of bin values and list of bin borders
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # are there multiple datasets?
    if not hasattr(data, 'values') and hasattr(data[0], '__len__'):
        multiple = len(data)
    else:
        multiple = 0

    if density:
        if multiple > 0:
            weights = list()
            for i in range(multiple):
                length = len(data[i]) - np.count_nonzero(np.isnan(data[i]))
                if length != len(data[i]):
                    logger.warning(f'Ignoring {len(data[i]) - length} NaNs for the density histogram.')
                weights.append(np.ones_like(data[i]) / float(length))
        else:
            weights = np.ones_like(data) / float(len(data))
    else:
        weights = None

    if n_bins is not None:
        if xlog:
            bin_borders = np.logspace(np.log10(xlim[0]), math.log10(xlim[1]), n_bins + 1)
        else:
            bin_borders = np.linspace(xlim[0], xlim[1], n_bins + 1)
    elif bin_borders is None:
        logger.warning('Neither a desired number of bins nor a list of bin borders was given.')

    if xlog:
        ax.set_xscale('log')

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if n_yticks is not None:
        if ylim is None:
            ylim = ax.get_ylim()
        yticks = np.linspace(ylim[0], ylim[1], n_yticks)
        ax.set_yticks(yticks)

    if xlog:
        ax.set_xlim((bin_borders[0], bin_borders[-1]))
    else:
        ax.set_xlim((bin_borders[0] - (xlim[1] - xlim[0]) * 0.01, bin_borders[-1] + (xlim[1] - xlim[0]) * 0.01))

    if xticklabels is not None:
        n_xticks = len(xticklabels)

    if n_xticks is not None:
        if not xlog:
            xticks = np.linspace(xlim[0], xlim[1], n_xticks)
        else:
            xticks = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), n_xticks)
        ax.set_xticks(xticks, minor=True if n_xticks > 10 else False)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if bin_weights is not None:
        if multiple == 0:
            # because all but the last (righthand-most) bin is half-open,
            # we need to manually exclude values exactly at the right edge
            counts, bins = np.histogram(data[data < xlim[1]], bins=bin_borders)
            data = bins[:-1]
            weights = counts * bin_weights
        else:
            weights = bin_weights

    bin_values, bin_borders, patches = plt.hist(data, bins=bin_borders, align=align, rwidth=rwidth, clip_on=clip_on,
                                                weights=weights, color=bar_color, alpha=alpha)

    if density and sum(bin_values) < 0.95:
        logger.warning(
            f'Although histogram density parameter is True, the sum of all bars is only {sum(bin_values):.3f}. '
            + f'Potentially due to the x-axis limits or NaN values in the input data.')

    if multiple == 0:
        logger.debug('Bin values and borders: '
                     + ', '.join(f'[{start:.2e},{end:.2e}]: {val:.1e}'
                                 for start, end, val in zip(bin_borders[:-1], bin_borders[1:], bin_values)))
    else:
        for values in bin_values:
            logger.debug('Bin values and borders: '
                         + ', '.join(f'[{start:.2e},{end:.2e}]: {val:.1e}'
                                     for start, end, val in zip(bin_borders[:-1], bin_borders[1:], values)))

    if highlight_patches is not None and len(highlight_patches) > 0:
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

    # add provided text notes at the given positions to the plot
    if notes is not None:
        add_notes(notes, ax, txt_color=bar_color, txt_fontsize=tick_fontsize)

    set_axis_style(ax, xlim, ylim, outward=axes_separation)

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)
        logger.info(f'Created histogram plot: {output_fp}')

    return bin_values, bin_borders


def plot_barplot(xss, yss, width=0.8, xlim=None, ylim=None, n_xticks=None, n_yticks=None, xticks=None, xticklabels=None,
                 align='center', xlog=False, ylog=False, xlabel=None, ylabel=None, title=None, axes_separation=5,
                 figsize=FIG_SIZE, bar_color='dimgrey', alpha=1.0, lbl_fontsize=LABEL_FS, tick_fontsize=TICK_FS,
                 notes=None, xs_line=None, ys_line=None, linewidth=1.0, linecolor='firebrick', linestyle='-',
                 ax=None, output_fp=None):
    """
    Create a bar plot, if xs and ys are multiple datasets then a grouped our stacked bar plot
    :param xss: array-like list of x positions
    :param yss: array-like list of y positions
    :param width: with of bars
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param n_xticks: number of tick marks on the x-axis (mutually-exclusive with xticks)
    :param n_yticks: number of tick marks on the y-axis
    :param xticks: array-like list of tick marks (mutually-exclusive with n_xticks)
    :param xticklabels: x-axis tick labels
    :param align: bin alignment (default: 'center')
    :param xlog: have x axis in logarithmic scale
    :param ylog: have y axis in logarithmic scale
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: plot title
    :param axes_separation: style parameter for the separation of x and y axes
    :param figsize: figure size given as a tuple
    :param bar_color: color of bars
    :param alpha: opacity of bars
    :param lbl_fontsize: font size of axis labels
    :param tick_fontsize: font size of tick marks
    :param notes: dictionary of dictionaries where the key is a tuple of x and y position of the text label and the
                  values specifies various other properties
    :param xs_line: array-like list of x-values
    :param ys_line: array-like list of y-values
    :param linewidth: line width
    :param linecolor: color of line of optional xy-line plot
    :param linestyle: style of line of optional xy-line plot
    :param ax: axes object of an already created figure
    :param output_fp: path to pdf output file
    :return: bar container
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if hasattr(yss[0], '__len__'):
        bar_container = []
        if hasattr(xss[0], '__len__'):
            for i, (xs, ys) in enumerate(zip(xss, yss)):
                bar_container.append(
                    ax.bar(xs, ys, width=width, align=align, color=bar_color[i], alpha=alpha, clip_on=False))
        else:
            bottom = np.zeros_like(yss[0])
            for i, ys in enumerate(yss):
                bar_container.append(ax.bar(xss, ys, bottom=bottom,
                                            width=width, align=align, color=bar_color[i], alpha=alpha, clip_on=False))
                bottom += ys

    else:
        bar_container = ax.bar(xss, yss, width=width, align=align, color=bar_color, alpha=alpha, clip_on=False)

    if xs_line is not None and ys_line is not None:
        ax.plot(xs_line, ys_line, lw=linewidth, clip_on=False,  # alpha=alpha,
                fillstyle='none', color=linecolor, linestyle=linestyle)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        xlim = ax.get_xlim()

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if xticklabels is not None:
        n_xticks = len(xticklabels)

    if n_xticks is not None and xticks is None:
        if not xlog:
            xticks = np.linspace(xlim[0], xlim[1], n_xticks)
        else:
            xticks = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), n_xticks)
        ax.set_xticks(xticks)
    elif n_xticks is None and xticks is not None:
        ax.set_xticks(xticks)
    elif n_xticks is not None and xticks is not None:
        logger.warning('Arguments n_xticks and xticks are mutually exclusive!')

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if n_yticks is not None:
        yticks = np.linspace(ylim[0], ylim[1], n_yticks)
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=lbl_fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=lbl_fontsize)

    set_axis_style(ax, xlim, ylim, outward=axes_separation)

    # change the fontsize of ticks labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    # add provided text notes at the given positions to the plot
    if notes is not None:
        add_notes(notes, ax, txt_color=bar_color, txt_fontsize=tick_fontsize)

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)
        logger.info(f'Created bar plot: {output_fp}')

    return bar_container


def plot_xy(xss, yss, xlim=None, ylim=None, legend=True, legend_loc='best', bbox_to_anchor=None, leg_ncol=1,
            xlog=False, ylog=False, n_xticks=None, n_yticks=None, sci_notation_axes=None, clip_on=False,
            x_offset_text_pos=None, y_offset_text_pos=None, lbl_fontsize=LABEL_FS,
            xlabel=None, ylabel=None, title=None, labels=None, colors=None, alpha=1.0, markers=None, markersizes=None,
            xticklabels=None, yticklabels=None, linestyles=None, linewidths=None,
            figsize=FIG_SIZE, notes=None, output_fp=None, ax=None):
    """
    Create xy line plot
    :param xss: array-like list of x positions or list of array-like list
    :param yss: list of array-like list of y positions
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
    :param n_yticks: number of tick marks on the y-axis
    :param sci_notation_axes: list of axes where scientific notation is enforced, e.g. ['x', 'y']
    :param clip_on: should lines be visible outside of the axes (default: False)
    :param x_offset_text_pos: change position of the x-axis exponent scientific notation label, e.g. (1.03, -0.05)
    :param y_offset_text_pos: change position of the y-axis exponent scientific notation label, e.g. (0.03, 1.05)
    :param lbl_fontsize: label font size
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: plot title
    :param labels: labels for the individual lines in the legend
    :param colors: list of colors for the individual lines
    :param alpha: transparency of lines between 0 and 1 (default 1), opacity of lines
    :param markers: list of markers for the individual lines
    :param markersizes: list of marker sizes
    :param xticklabels: list of xtick labels
    :param yticklabels: list of ytick labels
    :param linestyles: list of line styles for the individual lines (e.g., -, :-, --)
    :param linewidths: list of widths of the lines
    :param figsize: figure size given as a tuple
    :param notes: dictionary of dictionaries where the key is a tuple of x and y position of the text label and the
                  values specifies various other properties
    :param output_fp: path to pdf output file
    :param ax: axes object of an already created figure
    :return: list of Line2D objects
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    lines = []
    for i, ys in enumerate(yss):
        if hasattr(xss[0], '__len__'):
            xs = xss[i]
        else:
            xs = xss

        lines.append(ax.plot(xs, ys, alpha=alpha, clip_on=clip_on,
                             fillstyle='none', color='dimgrey' if colors is None else colors[i],
                             lw=1.3 if linewidths is None else linewidths[i],
                             linestyle='-' if linestyles is None else linestyles[i],
                             marker=None if markers is None else markers[i],
                             markersize=10 if markersizes is None else markersizes[i],
                             label=None if labels is None else labels[i]))

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)
        if n_xticks is not None:
            if not xlog:
                xticks = np.linspace(xlim[0], xlim[1], n_xticks)
            else:
                xticks = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), n_xticks)
            ax.set_xticks(xticks, minor=True if n_xticks > 10 else False)
    else:
        xlim = ax.get_xlim()

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if ylim is not None:
        ax.set_ylim(ylim)
        if n_yticks is not None:
            if not ylog:
                yticks = np.linspace(ylim[0], ylim[1], n_yticks)
            else:
                yticks = np.logspace(math.log10(ylim[0]), math.log10(ylim[1]), n_yticks)
            ax.set_yticks(yticks, minor=True if n_yticks > 10 else False)
    else:
        ylim = ax.get_ylim()

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # change given axis to scientific notation
    if sci_notation_axes is not None:
        for a in sci_notation_axes:
            ax.ticklabel_format(axis=a, style="sci", scilimits=(0, 0))

    if x_offset_text_pos is not None:
        # need to update the plot to be able to get the offset text
        plt.draw()
        x_offset_text = ax.xaxis.get_offset_text().get_text()
        ax.xaxis.get_offset_text().set_visible(False)
        ax.text(x_offset_text_pos[0], x_offset_text_pos[1], x_offset_text, fontsize=TICK_FS, transform=ax.transAxes)

    if y_offset_text_pos is not None:
        # need to update the plot to be able to get the offset text
        plt.draw()
        y_offset_text = ax.yaxis.get_offset_text().get_text()
        ax.yaxis.get_offset_text().set_visible(False)
        ax.text(y_offset_text_pos[0], y_offset_text_pos[1], y_offset_text, fontsize=TICK_FS, transform=ax.transAxes)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=lbl_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=lbl_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=lbl_fontsize)

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

    # add provided text notes at the given positions to the plot
    if notes is not None:
        add_notes(notes, ax)

    if output_fp is not None:
        plt.draw()
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)
        logger.info(f'Created xy line plot: {output_fp}')

    return lines


def set_axis_style(ax, xlim=None, ylim=None, outward=0):
    """
    Remove top and right axis of box around the plot
    Set end of x-axis and y-axis according to the given limits
    :param ax: axes object of an already created figure
    :param xlim: tuple defining limits of x-axis
    :param ylim: tuple defining limits of y-axis
    :param outward: amount of space to separate left and bottom axis
    :return:
    """

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim is None:
        ylim = ax.get_ylim()
    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    if xlim is None:
        xlim = ax.get_xlim()
    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

    for line in ['left', 'bottom']:
        ax.spines[line].set_position(('outward', outward))


def add_notes(notes, ax, txt_color='dimgrey', txt_fontsize=10):
    """
    Add provided text notes at the given positions to the plot
    :param notes: dictionary of dictionaries where the key is a tuple of x and y position of the text label and the
                  values specifies various other properties
    :param ax: axes object of an already created figure
    :param txt_color: color of text
    :param txt_fontsize: text fontsize
    """
    for (xpos, ypos), lbl_dict in notes.items():
        ax.text(
            xpos, ypos, lbl_dict['text'],
            bbox=lbl_dict['bbox'] if 'bbox' in lbl_dict else {'facecolor': 'none', 'edgecolor': 'none', 'alpha': 1.0},
            ha=lbl_dict['ha'] if 'ha' in lbl_dict else 'center',
            color=lbl_dict['color'] if 'color' in lbl_dict else txt_color,
            size=lbl_dict['fontsize'] if 'fontsize' in lbl_dict else txt_fontsize,
            transform=ax.transData if 'transform' in lbl_dict else ax.transAxes)


def add_second_xaxis(ax, xlog=False, xticks=None, xticklabels=None, xlabel=None, xlabel_coords=None, axes_separation=3,
                     output_fp=None):
    """
    Add a second x-axis at the top of the plot
    :param ax: axes object of an already created figure
    :param xlog: have x axis in logarithmic scale
    :param xticks: locations of xticks
    :param xticklabels: x-axis tick labels
    :param xlabel: label of x-axis
    :param xlabel_coords: location of xlabel
    :param axes_separation: style parameter for the separation of x and y axes
    :param output_fp: path to pdf output file
    """
    ax2 = ax.twiny()

    if xlog:
        ax2.set_xscale('log')

    ax2.set_xlim(ax.get_xlim())

    if xticks is not None:
        ax2.set_xticks(xticks)

    if xticklabels is not None:
        ax2.set_xticklabels(xticklabels)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.minorticks_off()
    for line in ['left', 'top']:
        ax2.spines[line].set_position(('outward', axes_separation))

    if xlabel is not None:
        ax2.set_xlabel(xlabel, fontsize=LABEL_FS)

    if xlabel_coords is not None:
        ax2.xaxis.set_label_coords(xlabel_coords[0], xlabel_coords[1])

    if output_fp is not None:
        plt.savefig(output_fp, dpi=150, bbox_inches='tight', transparent=True)
        logger.info(f'Saved plot with second x-axis: {output_fp}')
