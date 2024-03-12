"""
Plotting of seismic files.

The seisplot Python module provides basic functionality to display
seismic data in typical standard image or wiggle displays.

Author & Copyright: Dr. Thomas Hertweck, geophysics@email.de

License: GNU Lesser General Public License, Version 3
         https://www.gnu.org/licenses/lgpl-3.0.html
"""

__version__ = "1.0.1"
__author__ = "Thomas Hertweck"
__copyright__ = "(c) 2024 Thomas Hertweck"
__license__ = "GNU Lesser General Public License, Version 3"

from ._seisplt import SeisPlt


def plot(data, **kwargs):
    """
    Display seismic data in a highly configurable way.

    Parameters
    ----------
    data : Numpy structured array or Numpy array
        The seismic data to plot, either as Numpy structured array with
        trace headers, or as plane Numpy array (just the traces' amplitude
        values). The actual array with seismic amplitudes should have
        shape (ntraces, nsamples).
    fig : mpl.figure.Figure, optional (default: None)
        An existing Maplotlib figure to use. The default 'None' creates
        a new one.
    ax : mpl.axes.Axes, optional (default: None)
        An existing Matplotlib axes object to use for this plot. The
        default 'None' creates a new one.
    plottype : str, optional (default: 'image')
        The type of plot to create, either 'image' (default) or 'wiggle'.
    width : float, optional (default: 6)
        The width of the plot (inches).
    height : float, optional (default: 10)
        The height of the plot (inches).
    perc : float, optional (default: 100)
        The percentile to use when determining the clip values. The
        default uses all the data. The value of 'perc' must be in the
        range (0, 100].
    skip : int, optional (default: 1)
        For wiggle plots, the number of traces to skip to reduce the total
        number of traces to plot. Wiggle plots do not work well with a lot
        of traces to plot. If this value is greater than 1, every skip'th
        trace will be plotted instead of all the traces.
    xcur : float, optional (default: 1.0)
        For wiggle plots, the wiggle excursion in traces corresponding to
        the actual clip.
    normalize : str, optional (default: None)
        If set to 'trace', each trace will be normalized individually such
        that its maximum amplitude is one. If set to 'section', the
        entire section will be normalized such that its maximum is one.
        The default 'None' means no normalization is applied.
    lowclip : float, optional (default: None)
        Clip value at the lower end. Not to be used together with 'perc'.
        The default of 'None' means the lowest data value is used.
    highclip : float, optional (default: None)
        Clip value at the upper end. Not to be used together with 'perc'.
        Must be larger than 'lowclip' if both are given. The default of
        'None' means the highest data value is used.
    alpha : float, optional (default: 1.0)
        The transparency of image plots or wiggle fills. Must be between
        0 and 1. The default of 1 means no transparency.
    tight : bool, optional (default: True)
        Flag whether to apply matplotlib's tight layout.
    interpolation : str, optional (default: 'bilinear')
        The type of interpolation for image plots. See Matplotlib's
        documentation for valid strings.
    colormap : str, optional (default: 'seismic')
        The colormap for image plots. See Matplotlib's documentation for
        valid strings.
    linewidth : float, optional (default: 0.2)
        The width of lines in wiggle plots.
    linecolor : str, optional (default: 'black')
        The line color for wiggle plots.
    facecolor : str, optional (default: 'white')
        The background color of the actual plot area.
    wiggledraw : bool, optional (default: True)
        Whether to draw the wiggle trace.
    wigglefill : bool, optional (default: True)
        Whether to fill the wiggles. Setting both 'wiggledraw' and
        'wigglefill' to False leads to an empty plot.
    wigglehires : bool, optional (default: False)
        Whether to create an oversampled, high-resolution trace before
        plotting it in plottype 'wiggle'. This creates more accurate
        shading for filled wiggles.
    fillcolor : str, optional (default: 'black')
        The color with which wiggles will be filled.
    vaxis: numeric array, optional (default: None)
        The values for the vertical axis (typically 'time' or 'depth').
        If not set, the sample number might be used.
    vaxisbeg : float, optional (default: None)
        The first value to draw on the vertical axis. Defaults to the first
        value in 'vaxis' if 'None' is specified.
    vaxisend : float, optional (default: None)
        The last value to draw on the vertical axis. Defaults to the last
        value in 'vaxis' if 'None' is specified.
    vlabel : string, optional (default: None)
        Label on vertical axis.
    vlabelpos : string, optional  (default: 'center')
        Position of vertical label, 'bottom', 'top' or 'center'.
    haxis : numeric array or str, optional (default: None)
        The values for the horizontal axis. If given, the array will be
        used directly. If a string is given which should correspond to a
        trace header mnemonic, then the values will be taken from the
        ensemble's header table if available. As fallback, a simple
        trace number counter is used.
    haxisbeg : float, optional (default: None)
        The first value to draw on the horizontal axis. Defaults to the
        first value in 'haxis' if 'None' is specified.
    haxisend : float, optional (default: None)
        The last value to draw on the horizontal axis. Defaults to the
        last value in 'haxis' if 'None' is specified.
    hlabel : string, optional (default: None)
        Label on horizontal axis.
    hlabelpos : string, optional (default: 'center')
        Position of horizontal label, 'left', 'right' or 'center'.
    labelfontsize: int, optional (default: 12)
        The font size for labels.
    labelcolor: str, optional (default: 'black')
        The color to use for labels.
    vmajorticks: float, optional (default: None)
        The spacing at which to draw major ticks along the vertical axis.
        Defaults to Matplotlib's standard algorithm.
    vminorticks: float, optional (default: None)
        The spacing at which to draw minor ticks along the vertical axis.
        Must be smaller than 'vmajorticks'. Defaults to Matplotlib's
        standard behavior.
    hmajorticks: float, optional (default: None)
        The spacing at which to draw major ticks along the horizontal axis.
        Defaults to Matplotlib's standard algorithm.
    hminorticks: float, optional (default: None)
        The spacing at which to draw minor ticks along the horizontal axis.
        Must be smaller than 'hmajorticks'. Defaults to Matplotlib's
        standard behavior.
    majorticklength : float, optional (default: 6)
        The length of major ticks.
    minorticklength : float, optional (default: 4)
        The length of minor ticks.
    majortickwidth : float, optional (default: 1)
        The width of major ticks.
    minortickwidth : float, optional (default: 0.8)
        The width of minor ticks.
    ticklabelsize : int, optional (default: 10)
        The font size of tick labels.
    tickdirection : str, optional (default: 'out')
        Draw ticks to the outside ('out') or inside ('in').
    ticktop : boolean, optional (default: False)
        Draw ticks and horizontal label at the top (True) instead of bottom
        (False).
    vticklabelrot : float, optional (default: 0)
        Rotation angle of vertical tick labels (in degrees).
    hticklabelrot : float, optional (default: 0)
        Rotation angle of horizontal tick labels (in degrees).
    vtickformat : str, optional (default: None)
        The format to use for vertical tick labels. Defaults to
        Matplotlib's standard behavior.
    htickformat : str, optional (default: None)
        The format to use for horizontal tick labels. Defaults to
        Matplotlib's standard behavior.
    vgrid : str, optional (default: None)
        If 'None', no grid will be drawn. If set to 'major', a grid for
        major ticks will be drawn. If set to 'both', a grid for major
        and minor ticks will be drawn. This option sets grid lines for
        the vertical axis, i.e., they are displayed horizontally.
    hgrid : str, optional (default: None)
        If 'None', no grid will be drawn. If set to 'major', a grid for
        major ticks will be drawn. If set to 'both', a grid for major
        and minor ticks will be drawn. This option sets grid lines for
        the horizontal axis, i.e., they are displayed vertically.
    gridlinewidth : float, optional (default: 0.8)
        The linewidth of grid lines.
    gridlinealpha : float, optional (default: 0.5)
        The alpha (transparency) value for grid lines.
    gridstyle : str, optional (default: '-')
        The style of grid lines. Defaults to solid. See Matplotlib's
        documentation for valid options.
    gridcolor : str, optional (default: 'black')
        The color of grid lines.
    colorbar : bool, optional (default: False)
        Whether to draw a colorbar for image plots.
    colorbarlabel : str, optional (default: None)
        The label (typically indicating units) of the colorbar.
    colorbarshrink : float, optional (default: 0.4)
        The vertical scaling factor for the size of the colorbar.
    colorbarfraction: float, optional (default: 0.1)
        The horizontal fraction of the entire figure size that the colorbar
        may use. Default is 10%.
    colorbarpad : float, optional (default: 0.02)
        Padding between the figure and the colorbar. Defaults to 2%.
    colorbarlabelpad : float, optional (default: 0)
        Padding applied between the colorbar and the colorbarlabel.
    colorbarticklabelsize : int, optional (default: 10)
        The font size of colorbar tick labels.
    colorbarlabelsize : int, optional (default: 10)
        The font size of the colorbar label.
    colorbarbins : int, optional (default: None)
        The number of bins to use for determining colorbar ticks. The
        default of 'None' uses Matplotlib's standard behavior.
    title : str, optional (default: None)
        The title of the plot.
    titlefontsize : int, optional (default: 14)
        The fontsize for the title string.
    titlecolor : str, optional (default: 'black')
        The color used for the title.
    titlepos : str, optional (default: 'center')
        The position of the title, 'left', 'right', or 'center'.
    mnemonic_dt : str, optional (default: 'dt')
        The trace header mnemonic specifying the sampling interval. Only used
        when the traces are given as a Numpy structured array.
    mnemonic_delrt: str, optional (default: 'delrt')
        The trace header mnemonic specifying the delay recording time. Only
        used when the traces are given as a Numpy structured array.
    file : str, optional (default: None)
        Produce an output file on disk using the specified file name. The
        format of the output file is determined by the name's suffix.
    dpi : int (default: 'figure')
        The dots per inch to use for file output in non-vector graphics
        formats. The special value 'figure' (default) uses the figure's
        dpi value.

    Returns
    -------
    figure.Figure, axes.Axes
        Matplotlib's figure.Figure and axes.Axes object.
    """
    myplot = SeisPlt(data, **kwargs)
    return myplot.show()
