"""
Plotting of seismic files.

The seisplot Python module provides basic functionality to display
seismic data in typical standard image or wiggle displays.

Author & Copyright: Dr. Thomas Hertweck, geophysics@email.de

License: GNU Lesser General Public License, Version 3
         https://www.gnu.org/licenses/lgpl-3.0.html
"""

__version__ = "1.2.0"
__author__ = "Thomas Hertweck"
__copyright__ = "(c) 2025 Thomas Hertweck"
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
    label : str, optional (default: None)
        Label for potential legend of wiggle plots. Primarily useful if
        several wiggle plots are combined into one figure.
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
    ampfac : float, optional (default: 1.0)
        When plotting several wiggle plots in one figure, amplitude scaling
        factor to get relative wiggle excursions correct. Basically, the
        ratio between the maximum absolute amplitudes in both data sets.
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
    fillneg: bool, optional (default: False)
        If wigglefill is True, fill negative amplitude lobes instead of
        positive amplitude lobes.
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


def toggle(alldata, **kwargs):
    """
    Toggle seismic images.

    Basically, the parameters are identical to plot() parameters that hold
    for plottype='image'. In addition, there are three parameters to configure
    the toggle, namely 'interval', 'repeat_delay' and 'blit'. Note that saving
    the animation to a file is not available as part of this function - a user
    can simply run the save() method on the returned animation.ArtistAnimation
    object to output animated images (gif, apng) or movies (mp4, mkv).

    The toggle() function is typically used to display data before and after,
    for instance, a denoise process. A difference can also be included in the
    list of data sets to toggle. The display will automatically toggle, given
    a certain interval, between the different data sets in a loop.

    Parameters
    ----------
    alldata : list of Numpy structured arrays or Numpy arrays
        The seismic data to plot, either as Numpy structured arrays with
        trace headers, or as plane Numpy arrays (just the traces' amplitude
        values). The actual arrays with seismic amplitudes should have
        shape (ntraces, nsamples). At least two data sets have to be
        provided in the list. Information on axes etc. and data values is
        taken from the first data set, if applicable. The data sets must have
        the same size/shape.
    fig : mpl.figure.Figure, optional (default: None)
        An existing Maplotlib figure to use. The default 'None' creates
        a new one.
    ax : mpl.axes.Axes, optional (default: None)
        An existing Matplotlib axes object to use for this plot. The
        default 'None' creates a new one.
    interval : int, optional (default: 500)
        Delay between frames in milliseconds.
    repeat_delay : int, optional (default: 0)
        The delay in milliseconds between consecutive animation runs.
    blit : boolean, optional (default: False)
        Whether blitting is used to optimize drawing. With blit=True, the
        drawing will be faster. But some Artists like frame lines might
        get lost accidentally.
    width : float, optional (default: 6)
        The width of the plot (inches).
    height : float, optional (default: 10)
        The height of the plot (inches).
    perc : float, optional (default: 100)
        The percentile to use when determining the clip values. The
        default uses all the data. The value of 'perc' must be in the
        range (0, 100].
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
        The transparency of image plots. Must be between 0 and 1. The
        default of 1 means no transparency.
    tight : bool, optional (default: True)
        Flag whether to apply matplotlib's tight layout.
    interpolation : str, optional (default: 'bilinear')
        The type of interpolation for image plots. See Matplotlib's
        documentation for valid strings.
    colormap : str, optional (default: 'seismic')
        The colormap for image plots. See Matplotlib's documentation for
        valid strings.
    facecolor : str, optional (default: 'white')
        The background color of the actual plot area.
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

    Returns
    -------
    animation.ArtistAnimation, figure.Figure, axes.Axes
        Matplotlib's animation.ArtistAnimation, figure.Figure and axes.Axes object.
    """
    if not isinstance(alldata, list):
        raise TypeError("Need a list of at least two data sets.")
    if len(alldata) < 2:
        raise ValueError("Number of provided data sets is too small to animate.")

    plottype = "image"
    plottype = kwargs.pop("plottype", plottype).lower()
    if plottype != "image":
        raise ValueError("Toggles only work with plottype='image'.")

    interval = 500
    interval = kwargs.pop("interval", interval)
    if interval < 0:
        raise ValueError("Parameter 'interval' cannot be negative.")
    repeat_delay = 0
    repeat_delay = kwargs.pop("repeat_delay", repeat_delay)
    if repeat_delay < 0:
        raise ValueError("Parameter 'repeat_delay' cannot be negative.")
    blit = False
    blit = kwargs.pop("blit", blit)

    myplot = SeisPlt(alldata[0], plottype="image", **kwargs)
    return myplot._toggle(alldata, interval=interval, repeat_delay=repeat_delay,
                          blit=blit)

def wipe(data1, data2, **kwargs):
    """
    Wipe two images.

    Basically, the parameters are identical to plot() parameters that hold
    for plottype='image'. In addition, there are several parameters to
    configure the wiping process. Note that saving the animation to a file is
    not available as part of this function - a user can simply run the save()
    method on the returned animation.ArtistAnimation object to output animated
    images (gif, apng) or movies (mp4, mkv).

    The wipe() function is typically used to display data before and after a
    certain processing step.

    Parameters
    ----------
    data1 : Numpy structured array or Numpy array
        First seismic data to plot, either as Numpy structured array with
        trace headers, or as plane Numpy array (just the traces' amplitude
        values). Information on axes etc. and data values is taken from
        this first data set, if applicable.
    data2 : Numpy structured array or Numpy array
        Second seismic data to plot, either as Numpy structured array with
        trace headers, or as plane Numpy array (just the traces' amplitude
        values). The data set must have the same size/shape as data1.
    fig : mpl.figure.Figure, optional (default: None)
        An existing Maplotlib figure to use. The default 'None' creates
        a new one.
    ax : mpl.axes.Axes, optional (default: None)
        An existing Matplotlib axes object to use for this plot. The
        default 'None' creates a new one.
    interval : int, optional (default: 500)
        Delay between frames in milliseconds.
    repeat_delay : int, optional (default: 0)
        The delay in milliseconds between consecutive animation runs.
    blit : boolean, optional (default: False)
        Whether blitting is used to optimize drawing. With blit=True, the
        drawing will be faster. But some Artists like frame lines might
        get lost accidentally.
    nwipe : int, optional (default: 10)
        The number of steps to complete a full wiping process, i.e., to
        completely go from one image to the other image.
    direction : str, optional (default: 'lr')
        Whether to wipe from left to right and back ('lr') or from top to
        bottom and back,, i.e., up and down ('ud').
    drawwipe : boolean, optional (default: True)
        Whether to draw a line on each frame marking the current separation
        between the two images. The line will be vertical for 'lr' wipes and
        horizontal for 'ud' wipes.
    wipecolor : string, optional (default: 'black')
        The color of the line separating the two images on each frame.
    width : float, optional (default: 6)
        The width of the plot (inches).
    height : float, optional (default: 10)
        The height of the plot (inches).
    perc : float, optional (default: 100)
        The percentile to use when determining the clip values. The
        default uses all the data. The value of 'perc' must be in the
        range (0, 100].
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
        The transparency of image plots. Must be between 0 and 1. The
        default of 1 means no transparency.
    tight : bool, optional (default: True)
        Flag whether to apply matplotlib's tight layout.
    interpolation : str, optional (default: 'bilinear')
        The type of interpolation for image plots. See Matplotlib's
        documentation for valid strings.
    colormap : str, optional (default: 'seismic')
        The colormap for image plots. See Matplotlib's documentation for
        valid strings.
    facecolor : str, optional (default: 'white')
        The background color of the actual plot area.
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

    Returns
    -------
    animation.ArtistAnimation, figure.Figure, axes.Axes
        Matplotlib's animation.ArtistAnimation, figure.Figure and axes.Axes object.
    """
    plottype = "image"
    plottype = kwargs.pop("plottype", plottype).lower()
    if plottype != "image":
        raise ValueError("Wipes only work with plottype='image'.")

    interval = 500
    interval = kwargs.pop("interval", interval)
    if interval < 0:
        raise ValueError("Parameter 'interval' cannot be negative.")
    repeat_delay = 0
    repeat_delay = kwargs.pop("repeat_delay", repeat_delay)
    if repeat_delay < 0:
        raise ValueError("Parameter 'repeat_delay' cannot be negative.")
    nwipe = 10
    nwipe = kwargs.pop("nwipe", nwipe)
    if nwipe < 1:
        raise ValueError("Parameter 'nwipe' must be greater than 0.")
    direction = "lr"
    direction = kwargs.pop("direction", direction)
    if direction not in ["lr", "ud"]:
        raise ValueError("Parameter 'direction' must be 'lr' or 'ud'.")
    blit = False
    blit = kwargs.pop("blit", blit)
    drawwipe = True
    drawwipe = kwargs.pop("drawwipe", drawwipe)
    wipecolor = "black"
    wipecolor = kwargs.pop("wipecolor", wipecolor)

    myplot = SeisPlt(data1, plottype="image", **kwargs)
    return myplot._wipe(data1, data2, nwipe=nwipe, direction=direction,
                        interval=interval, repeat_delay=repeat_delay,
                        blit=blit, drawwipe=drawwipe, wipecolor=wipecolor)
