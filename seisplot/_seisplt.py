"""SeisPLT module to display seismic data."""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from collections.abc import Callable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import Normalize

import matplotlib.animation as animation

try:
    from numba import jit
except ImportError:
    # log.warning("Numba not installed. Using non-optimized code.")
    def jit(*args, **kwargs):
        """Create dummy decorator."""
        def decorator(func):
            return func
        return decorator


log = logging.getLogger(__name__)


def _is_integer(buffer):
    """
    Check whether a number is integer, or numbers in an array are all integers.

    Parameters
    ----------
    buffer : numeric
        Number or array of numbers to check.

    Returns
    -------
    boolean
        True if all numbers are integers, otherwise False.
    """
    return np.equal(np.mod(buffer, 1), 0).all()


@dataclass
class _PlotPara():
    """Dataclass to store plotting-related parameters."""

    plottype: str = "image"
    fig: mpl.figure.Figure = None
    ax: mpl.axes.Axes = None
    ax_ovl: mpl.axes.Axes = None
    style: str = "bmh"
    width: float = 6
    height: float = 10
    perc: float = 100.0
    skip: int = 1
    xcur: float = 1.0
    ampfac: float = 1.0
    normalize: str = None
    lowclip: float = None
    highclip: float = None
    alpha: float = 1.0
    tight: bool = True
    interpolation: str = "bilinear"
    colormap: str = None
    linewidth: float = None
    linecolor: str = "black"
    facecolor: str = "white"
    wiggledraw: bool = True
    wigglefill: bool = True
    wigglehires: bool = False
    fillcolor: str = "black"
    fillneg: bool = False
    vaxis: np.array = None
    vaxisbeg: float = None
    vaxisend: float = None
    vlabel: str = None
    vlabelpos: str = "center"
    haxis: np.array = None
    haxisbeg: float = None
    haxisend: float = None
    hlabel: str = None
    hlabelpos: str = "center"
    labelfontsize: int = 12
    labelcolor: str = "black"
    vmajorticks: float = None
    vminorticks: float = None
    hmajorticks: float = None
    hminorticks: float = None
    majorticklength: float = 6
    minorticklength: float = 4
    majortickwidth: float = 1
    minortickwidth: float = 0.8
    ticklabelsize: float = 10
    tickdirection: str = "out"
    ticktop: bool = False
    vticklabelrot: float = None
    hticklabelrot: float = None
    vtickformat: str = None
    htickformat: str = None
    vgrid: str = None
    hgrid: str = None
    gridlinewidth: float = 0.8
    gridlinealpha: float = 0.5
    gridstyle: str = "-"
    gridcolor: str = "black"
    colorbar: bool = False
    colorbarlabel: str = None
    colorbarshrink: float = None
    colorbarfraction: float = 0.1
    colorbarpad: float = 0.02
    colorbarlabelpad: float = 0
    colorbarlabelsize: float = 10
    colorbarticklabelsize: float = 10
    colorbarbins: int = None
    title: str = None
    titlefontsize: int = 14
    titlecolor: str = "black"
    titlepos: str = "center"
    vmm: float = 0
    mnemonic_dt: str = "dt"
    mnemonic_delrt: str = "delrt"
    file: str = None
    dpi: str = "figure"
    label: str = None
    ampspec: bool = False
    phaspec: bool = False
    window: Callable[[np.typing.ArrayLike], np.typing.ArrayLike] | np.typing.ArrayLike = None
    nfft: int = None
    scale: str = "linear"
    unwrap: bool = True
    degree: bool = False
    fftnorm: str = "backward"
    smooth: bool = False
    smoothwindow: float = None
    overlay: str = None
    overlaycolor: str = "red"
    overlaylabel: str = None
    overlaylinewidth: float = 0.8
    ovlmajorticks: float = None
    ovlminorticks: float = None
    ovlaxisbeg: float = None
    ovlaxisend: float = None
    overlayinvert: bool = False
    hkey: str = None
    hinvert: bool = False
    hscale: float = 1.0
    vkey: str = None
    vscale: float = 1.0
    vinvert: bool = False
    color_by: str = "header"
    ckey: str = None
    size_by: str = None
    size_scale: float = 1.0
    marker: str = "o"
    markersize: float = 20
    edgecolor: str = "face"
    equal_axes: bool = False
    color_norm: str = "linear"
    histogram: bool = False
    histogram_colorize: bool = True
    histogram_color: str ="gray"
    histogram_vpos: float = 1.08
    histogram_height: float = 0.18
    histogram_hlabel: str = "auto"
    histogram_vlabel: str = "auto"
    histogram_density: bool = False
    histogram_bins: str = "auto"


class SeisPlt():
    """Class to handle seismic data displays."""

    def __init__(self, data, **kwargs):
        """
        Display seismic data.

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
        style : str, optional (default: 'bmh')
            The style sheet to use; only set in case a new figure is created.
            Should not be used explicitly. Use a style context manager.
        plottype : str, optional (default: 'image')
            The type of plot to create, either 'image' (default) or 'wiggle'
            or 'spectrum'.
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
            The transparency of image plots or wiggle fills or crossplot fills.
            Must be between 0 and 1. The default of 1 means no transparency.
        tight : bool, optional (default: True)
            Flag whether to apply matplotlib's tight layout.
        interpolation : str, optional (default: 'bilinear')
            The type of interpolation for image plots. See Matplotlib's
            documentation for valid strings.
        colormap : str, optional (default: 'seismic' or 'jet')
            The colormap for image plots (default: 'seismic') or crossplots
            (default: 'jet'). See Matplotlib's documentation for valid strings.
        linewidth : float, optional (default: 0.2 or 0)
            The width of lines in wiggle plots (default: 0.2). Or the linewidth
            of edges in crossplots (default: 0).
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
        hticklabelrot : float, optional (default: None, i.e., 0)
            Rotation angle of horizontal tick labels (in degrees).
        vtickformat : str, optional (default: None, i.e., 0)
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
        colorbarshrink : float, optional (default: 0.4 or 1.0)
            The vertical scaling factor for the size of the colorbar for image
            plots (default 0.4) or for crossplots (default: 1.0).
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
        colorbarbins : int,. optional (default: None)
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
        overlay : str, optional (default: None)
            The trace header mnemonic to plot on top of the data. Only works if
            data are provided as structured array. In addition to trace header
            mnemonics, the special keyword 'rms' can be supplied which plots
            each traces RMS amplitude value on top of the data display (also
            works if data are provided as unstructured array). This only makes
            real sense if seismic data are plotted (as opposed to, for instance,
            velocity fields). If used with colorbar=True, you probably need to
            increase colorbarpad as the colorbar will otherwise interfere with
            the overlay axis.
        overlaycolor : str, optional (default: 'red')
            The color of the overlay plot.
        overlaylabel : str, optional (default: None)
            Label on vertical overlay axis.
        overlaylinewidth : float, optional (default: 0.8)
            Linewidth of overlay display.
        overlaymajorticks: float, optional (default: None)
            The spacing at which to draw major ticks along the vertical
            overlay axis. Defaults to Matplotlib's standard algorithm.
        overlayminorticks: float, optional (default: None)
            The spacing at which to draw minor ticks along the vertical
            overlay axis. Must be smaller than 'overlaymajorticks'. Defaults
            to Matplotlib's standard behavior.
        overlayaxisbeg : float, optional (default: None)
            The first value to draw on the vertical overlay axis. Defaults to
            the smallest value of the quantity to be plotted if 'None'.
        overlayaxisend : float, optional (default: None)
            The last value to draw on the vertical overlay axis. Defaults to
            the largest value  of the quantity to be plotted if 'None'.
        overlayinvert: bool, optional (default: False)
            Invert the vertical overlay axis.
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
        amplitude : bool (default: False)
            Whether to plot the amplitude spectrum. One of amplitude or phase
            must be set to True.
        phase : bool (default: False)
            Whether to plot the phase spectrum. One of amplitude or phase must
            be set to True.
        window : callable or Numpy array, optional (default: None)
            A function or a vector of length nsamples used to window the data
            before performing a Fourier transform, typically used to taper the
            traces at their beginning and ending. For instance, you could use
            'window=np.hanning' to apply a Hanning window to the traces. The
            function must be callable with a single argument, the number of
            samples.
        nfft : int, optional (default: nsamples)
            The Fourier transform length. By default, the number of samples is
            used. If nfft is larger, then zeros will be padded. If nfft is
            smaller, then traces will be truncated.
        fftnorm : str, optional (default: 'backward')
            Where to apply the FFT normalization factor. The default 'backward'
            applies no scaling in the forward transform. The alternative
            'forward' applies the full scaling in the forward transform, and
            'ortho' applies 1/sqrt(nfft) on both the forward and backward.
            transform. Note that there is always a scaling factor of 2 in the
            amplitude spectrum as only positive frequencies are displayed.
        scale : str, optional (default 'linear')
            Applies to amplitude spectra only. Whether to plot a linear amplitude
            spectrum (default), or an amplitude spectrum in dezibel ('dB').
        unwrap : bool, optional (default: True)
            Applies to phase spectra only. Whether to unwrap the phase spectrum
            or not.
        degree : bool, optional (default: False)
            Applies to phase spectra only. Whether to display the phase in
            degrees or not; default is a display in radians.
        smooth : bool, optional (default: False)
            Smooth the amplitude spectrum using moving average.
        smoothwindow : float, optional (default: 5*df)
            The length of the moving average window for smoothing (in Hz).
        hkey : str (default: None)
            Header mnemonic to define the horizontal axis of crossplots.
        vkey : str (default: None)
            Header mnemonic to define the vertical axis of crossplots.
        hscale : float, optional (default: 1.0)
            Scaling factor to apply to the values of hkey.
        vscale : float, optional (default: 1.0)
            Scaling factor to apply to the values of vkey.
        hinvert : bool, optional (default: False)
            Invert the horizontal axis if True.
        vinvert : bool, optional (default: False)
            Invert the vertical axis (default: False)
        color_by : str, optional (default: None)
            How to color the points of the crossplot. Either 'None' to use
            Matplotlib's default, or the name of a single color (e.g., 'red'),
            or one of the following strings:
            'header' to color the points based on values of another header
            mnemonic; 'rms' to color each point by the RMS amplitude value of
            the corresponding trace; 'mean' to color by the average value of
            a trace (this allows you to easily spot a trace bias), 'max' to
            color by a trace's maximum value, 'min' to color by a trace's
            minimum value, 'median' to color by a trace's median amplitude,
            or 'fpeak' to color by the peak frequency of the corresponding
            trace. Obviously, all data-driven color approaches require the
            actual data to be available, not just pure trace header keys. If
            color_by is "array", then values can be passed as array using the
            parameter 'ckey'.
        ckey : str or Numpy array, optional (default: None)
            If color_by is 'header', then ckey specifies the header mnemonic
            to use to color the crossplot points. If color_by is "array", then
            ckey specifies the values to use to colorize the data. The array
            must have a length consistent with the number of traces in data.
        color_norm : str, optional (default: 'linear')
            The normalization method used to scale scalar data to the [0, 1]
            range before mapping to colors. Typically a scale name like, for
            instance, 'linear', 'log', etc. - see Matplotlib's documenatation
            for valid entries.
        size_by : str, optional (default: None)
            The size of each marker can be based on values of another header
            mnemonic. If so, specify the header mnemonic here.
        size_scale : float (default: 1.0)
            The marker size specified by the size_by header is in points**2.
            The parameter size_scale can be used to scale the values of the
            header mnemonic 'size_by' to a suitable range.
        marker : str, optional (default: 'o')
            The marker style. See Matplotlib's documentation for valid entries.
        markersize : float, optional (default: None)
            If a constant marker size is used rather than a variable size based
            on another header mnemonic (see parameter 'size_by'), the this
            parameter specifies the size of each marker in points**2.
        edgecolor : str, optional (default: 'face')
            The edge color of the marker. By default the edge color is the same
            as the face color.
        equal_axes : bool, optional (default: False)
            Whether to plot horizontal and vertical axes constrained (True) or
            not consrrained (False).
        histogram : bool, optional (default: False)
            Draw a histogram of the color_by values at the top of the crossplot.
        histogram_colorize : bool, optional (default: True)
            Colorize the histogram such that each histogram bar has the same
            color as the value on the chosen colormap.
        histogram_color : str, optional (default: "gray")
            The color of the histogram if histogram_colorize is False.
        histogram_vpos : float, optional (default: 1.08)
            The vertical position of the histogram in mpl.Axes coordinates.
            The default of 1.08 leaves a small gap and puts the histogram at
            the top of the crossplot.
        histogram_height : float, optional (default: 0.18)
            The height of the histogram in terms of mpl.Axes coordinates. The
            default makes the histogram about a fifth in height compared to the
            crossplot.
        histogram_hlabel : str, optional (default: "auto")
            The label at the horizontal axis of the histogram. The default
            "auto" chooses the label automatically. Set to None in order to
            have no label at all.
        histogram_vlabel : str, optional (default: "auto")
            The label at the vertical axis of the histogram. The default
            "auto" chooses the label automatically. Set to None in order to
            have no label at all.
        histogram_density : bool, optional (default: False)
            If set to True, the histogram contains the probability density at
            each bin rather than a count.
        histogram_bins : int or str, optional (default: "auto")
            The number of bins. Either an integer specifying the number of bins,
            or the string "auto" to have the number of bins determined
            automatically. See numpy.histogram for details on how this is done.
        """
        self._ensemble = data
        self._is_structured = False
        if isinstance(self._ensemble, np.ndarray):
            if self._ensemble.dtype.names is not None:
                self._is_structured = True
        self._par = _PlotPara()
        self._data = None
        self._parse_para(**kwargs)

    @staticmethod
    def tight():
        """Enable tight layout for figure."""
        plt.tight_layout()

    def _parse_para(self, **kwargs):
        """Parse parameters and initialize variables of PlotPara."""
        self._par.plottype = kwargs.pop("plottype", self._par.plottype).lower()
        if self._par.plottype not in ["image", "wiggle", "spectrum", "crossplot"]:
            raise ValueError(f"Unknown value '{self._par.plottype}' for parameter 'plottype'.")

        self._par.fig = kwargs.pop("fig", self._par.fig)
        self._par.ax = kwargs.pop("ax", self._par.ax)
        self._par.style = kwargs.pop("style", self._par.style)
        self._par.width = kwargs.pop("width", self._par.width)
        self._par.height = kwargs.pop("height", self._par.height)
        self._par.label = kwargs.pop("label", self._par.label)

        self._par.perc = kwargs.pop("perc", self._par.perc)
        if self._par.perc <= 0 or self._par.perc > 100:
            raise ValueError("Parameter 'perc' must be in range (0, 100.0].")

        self._par.skip = kwargs.pop("skip", self._par.skip)
        if self._par.skip < 1:
            raise ValueError("Parameter 'skip' must be >= 1.")

        self._par.xcur = kwargs.pop("xcur", self._par.xcur)
        if self._par.xcur <= 0.0:
            raise ValueError("Parameter 'xcur' must be > 0.")

        self._par.normalize = kwargs.pop("normalize", self._par.normalize)
        if self._par.normalize is not None:
            self._par.normalize = self._par.normalize.lower()
            if self._par.normalize not in ["trace", "section"]:
                raise ValueError(f"Unknown value {self._par.normalize} for parameter 'normalize'.")

        self._par.lowclip = kwargs.pop("lowclip", self._par.lowclip)
        self._par.highclip = kwargs.pop("highclip", self._par.highclip)
        self._par.ampfac = kwargs.pop("ampfac", self._par.ampfac)
        if self._par.ampfac == 0:
            raise ValueError("Parameter 'ampfac' cannot be zero.")
        self._par.alpha = kwargs.pop("alpha", self._par.alpha)
        self._par.tight = kwargs.pop("tight", self._par.tight)
        self._par.interpolation = kwargs.pop("interpolation", self._par.interpolation)
        self._par.colormap = kwargs.pop("colormap", self._par.colormap)
        self._par.linewidth = kwargs.pop("linewidth", self._par.linewidth)
        self._par.linecolor = kwargs.pop("linecolor", self._par.linecolor)
        self._par.facecolor = kwargs.pop("facecolor", self._par.facecolor)
        self._par.wiggledraw = kwargs.pop("drawwiggles", self._par.wiggledraw)
        self._par.wigglefill = kwargs.pop("wigglefill", self._par.wigglefill)
        self._par.wigglehires = kwargs.pop("wigglehires", self._par.wigglehires)
        self._par.fillcolor = kwargs.pop("fillcolor", self._par.fillcolor)
        self._par.fillneg = kwargs.pop("fillneg", self._par.fillneg)
        self._par.vaxis = kwargs.pop("vaxis", self._par.vaxis)
        self._par.vaxisbeg = kwargs.pop("vaxisbeg", self._par.vaxisbeg)
        self._par.vaxisend = kwargs.pop("vaxisend", self._par.vaxisend)
        self._par.vlabel = kwargs.pop("vlabel", self._par.vlabel)
        self._par.vlabelpos = kwargs.pop("vlabelpos", self._par.vlabelpos).lower()
        if self._par.vlabelpos not in ["bottom", "top", "center"]:
            raise ValueError(f"Unknown value '{self._par.vlabelpos}' for parameter 'vlabelpos'.")
        self._par.haxis = kwargs.pop("haxis", self._par.haxis)
        self._par.haxisbeg = kwargs.pop("haxisbeg", self._par.haxisbeg)
        self._par.haxisend = kwargs.pop("haxisend", self._par.haxisend)
        self._par.hlabel = kwargs.pop("hlabel", self._par.hlabel)
        self._par.hlabelpos = kwargs.pop("hlabelpos", self._par.hlabelpos).lower()
        if self._par.hlabelpos not in ["left", "right", "center"]:
            raise ValueError(f"Unknown value '{self._par.hlabelpos}' for parameter 'hlabelpos'.")
        self._par.labelfontsize = kwargs.pop("labelfontsize", self._par.labelfontsize)
        self._par.labelcolor = kwargs.pop("labelcolor", self._par.labelcolor)
        self._par.vmajorticks = kwargs.pop("vmajorticks", self._par.vmajorticks)
        self._par.vminorticks = kwargs.pop("vminorticks", self._par.vminorticks)
        self._par.hmajorticks = kwargs.pop("hmajorticks", self._par.hmajorticks)
        self._par.hminorticks = kwargs.pop("hminorticks", self._par.hminorticks)
        self._par.majorticklength = kwargs.pop("majorticklength", self._par.majorticklength)
        self._par.minorticklength = kwargs.pop("minorticklength", self._par.minorticklength)
        self._par.majortickwidth = kwargs.pop("majortickwidth", self._par.majortickwidth)
        self._par.minortickwidth = kwargs.pop("minortickwidth", self._par.minortickwidth)
        self._par.ticklabelsize = kwargs.pop("ticklabelsize", self._par.ticklabelsize)
        self._par.tickdirection = kwargs.pop("tickdirection", self._par.tickdirection).lower()
        if self._par.tickdirection not in ["out", "in"]:
            raise ValueError(f"Unknown value '{self._par.tickdirection}' "
                             "for parameter 'tickdirection'.")
        self._par.ticktop = kwargs.pop("ticktop", self._par.ticktop)
        self._par.vticklabelrot = kwargs.pop("vticklabelrot", self._par.vticklabelrot)
        self._par.hticklabelrot = kwargs.pop("hticklabelrot", self._par.hticklabelrot)
        self._par.vtickformat = kwargs.pop("vtickformat", self._par.vtickformat)
        self._par.htickformat = kwargs.pop("htickformat", self._par.htickformat)
        self._par.vgrid = kwargs.pop("vgrid", self._par.vgrid)
        self._par.hgrid = kwargs.pop("hgrid", self._par.hgrid)
        self._par.gridlinewidth = kwargs.pop("gridlinewidth", self._par.gridlinewidth)
        self._par.gridlinealpha = kwargs.pop("gridlinealpha", self._par.gridlinealpha)
        self._par.gridstyle = kwargs.pop("gridstyle", self._par.gridstyle)
        self._par.gridcolor = kwargs.pop("gridcolor", self._par.gridcolor)
        self._par.colorbar = kwargs.pop("colorbar", self._par.colorbar)
        self._par.colorbarlabel = kwargs.pop("colorbarlabel", self._par.colorbarlabel)
        self._par.colorbarshrink = kwargs.pop("colorbarshrink", self._par.colorbarshrink)
        self._par.colorbarfraction = kwargs.pop("colorbarfraction", self._par.colorbarfraction)
        self._par.colorbarpad = kwargs.pop("colorbarpad", self._par.colorbarpad)
        self._par.colorbarlabelpad = kwargs.pop("colorbarlabelpad", self._par.colorbarlabelpad)
        self._par.colorbarticklabelsize = kwargs.pop("colorbarticklabelsize",
                                                     self._par.colorbarticklabelsize)
        self._par.colorbarlabelsize = kwargs.pop("colorbarlabelsize", self._par.colorbarlabelsize)
        self._par.colorbarbins = kwargs.pop("colorbarbins", self._par.colorbarbins)
        self._par.title = kwargs.pop("title", self._par.title)
        self._par.titlefontsize = kwargs.pop("titlefontsize", self._par.titlefontsize)
        self._par.titlecolor = kwargs.pop("titlecolor", self._par.titlecolor)
        self._par.titlepos = kwargs.pop("titlepos", self._par.titlepos).lower()
        if self._par.titlepos not in ["left", "right", "center"]:
            raise ValueError(f"Unknown value '{self._par.titlepos}' for parameter 'titlepos'.")
        self._par.mnemonic_dt = kwargs.pop("mnemonic_dt", self._par.mnemonic_dt)
        self._par.mnemonic_delrt = kwargs.pop("mnemonic_delrt", self._par.mnemonic_delrt)
        self._par.file = kwargs.pop("file", self._par.file)
        self._par.dpi = kwargs.pop("dpi", self._par.dpi)
        if self._par.dpi != "figure" and self._par.dpi < 72:
            raise ValueError(f"Value ({self._par.dpi}) too small for parameter 'dpi'.")
        self._par.overlay = kwargs.pop("overlay", self._par.overlay)
        self._par.overlaycolor = kwargs.pop("overlaycolor", self._par.overlaycolor)
        self._par.overlaylabel = kwargs.pop("overlaylabel", self._par.overlaylabel)
        self._par.overlaylinewidth = kwargs.pop("overlaylinewidth", self._par.overlaylinewidth)
        self._par.ovlmajorticks = kwargs.pop("overlaymajorticks", self._par.ovlmajorticks)
        self._par.ovlminorticks = kwargs.pop("overlayminorticks", self._par.ovlminorticks)
        self._par.ovlaxisbeg = kwargs.pop("overlayaxisbeg", self._par.ovlaxisbeg)
        self._par.ovlaxisend = kwargs.pop("overlayaxisend", self._par.ovlaxisend)
        self._par.overlayinvert = kwargs.pop("overlayinvert", self._par.overlayinvert)
        self._par.smooth = kwargs.pop("smooth", self._par.smooth)
        self._par.smoothwindow = kwargs.pop("smoothwindow", self._par.smoothwindow)
        self._par.window = kwargs.pop("window", self._par.window)

        if self._par.plottype == "spectrum":
            if self._par.linewidth is None:
                self._par.linewidth = 1
            self._par.ampspec = kwargs.pop("amplitude", self._par.ampspec)
            self._par.phaspec = kwargs.pop("phase", self._par.phaspec)
            self._par.nfft = kwargs.pop("nfft", self._par.nfft)
            self._par.unwrap = kwargs.pop("unwrap", self._par.unwrap)
            self._par.scale = kwargs.pop("scale", self._par.scale).lower()
            self._par.fftnorm = kwargs.pop("fftnorm", self._par.fftnorm).lower()
            if self._par.fftnorm not in ["backward", "forward", "ortho"]:
                raise ValueError(f"Parameter 'fftnorm' ({self._par.fftnorm}) must be 'backward', 'forward' or 'ortho.")
            self._par.degree = kwargs.pop("degree", self._par.degree)
            if self._par.scale not in ["linear", "db"]:
                raise ValueError(f"Parameter 'scale' ({self._par.scale}) must be 'linear' or 'dB'.")
            if self._par.ampspec is False and self._par.phaspec is False:
                raise ValueError("Both parameters 'amplitude' and 'phase' are False, nothing to draw.")
            elif self._par.ampspec and self._par.phaspec:
                raise ValueError("Both parameters 'amplitude' and 'phase' are True - reset one of them.")
            if self._par.ampspec:
                if self._par.scale == "linear" and self._par.vaxisend is None:
                    self._par.vaxisend = 0
                elif self._par.scale == "db" and self._par.vaxisbeg is None:
                    self._par.vaxisbeg = 0
        elif self._par.plottype == "crossplot":
            if self._par.linewidth is None:
                self._par.linewidth = 0
            if self._par.colormap is None:
                self._par.colormap = "jet"
            if self._par.colorbarshrink is None:
                self._par.colorbarshrink = 1.0
            self._par.hkey = kwargs.pop("hkey", self._par.hkey)
            if isinstance(self._par.hkey, str):
                self._par.hkey = self._par.hkey.lower()
            if self._par.hkey is None:
                raise ValueError("Need a header mnemonic 'hkey' to define the horizontal axis.")
            self._par.vkey = kwargs.pop("vkey", self._par.vkey)
            if isinstance(self._par.vkey, str):
                self._par.vkey = self._par.vkey.lower()
            if self._par.vkey is None:
                raise ValueError("Need a header mnemonic 'vkey' to define the vertical axis.")
            self._par.hscale = kwargs.pop("hscale", self._par.hscale)
            self._par.vscale = kwargs.pop("vscale", self._par.vscale)
            self._par.color_by = kwargs.pop("color_by", self._par.color_by)
            if isinstance(self._par.color_by, str):
                self._par.color_by = self._par.color_by.lower()
            self._par.ckey = kwargs.pop("ckey", self._par.ckey)
            if isinstance(self._par.ckey, str):
                self._par.ckey = self._par.ckey.lower()
            self._par.size_by = kwargs.pop("size_by", self._par.size_by)
            if isinstance(self._par.size_by, str):
                self._par.size_by = self._par.size_by.lower()
            self._par.size_scale = kwargs.pop("size_scale", self._par.size_scale)
            self._par.marker = kwargs.pop("marker", self._par.marker)
            self._par.markersize = kwargs.pop("markersize", self._par.markersize)
            self._par.edgecolor = kwargs.pop("edgecolor", self._par.edgecolor)
            self._par.equal_axes = kwargs.pop("equal_axes", self._par.equal_axes)
            self._par.color_norm = kwargs.pop("color_norm", self._par.color_norm)
            self._par.hinvert = kwargs.pop("hinvert", self._par.hinvert)
            self._par.vinvert = kwargs.pop("vinvert", self._par.vinvert)
            self._par.histogram = kwargs.pop("histogram", self._par.histogram)
            self._par.histogram_colorize = kwargs.pop("histogram_colorize", self._par.histogram_colorize)
            self._par.histogram_color = kwargs.pop("histogram_color", self._par.histogram_color)
            self._par.histogram_vpos = kwargs.pop("histogram_vpos", self._par.histogram_vpos)
            self._par.histogram_height = kwargs.pop("histogram_height", self._par.histogram_height)
            self._par.histogram_hlabel = kwargs.pop("histogram_hlabel", self._par.histogram_hlabel)
            self._par.histogram_vlabel = kwargs.pop("histogram_vlabel", self._par.histogram_vlabel)
            self._par.histogram_bins = kwargs.pop("histogram_bins", self._par.histogram_bins)
            self._par.histogram_density = kwargs.pop("histogram_density", self._par.histogram_density)
        else:
            if self._par.linewidth is None:
                self._par.linewidth = 0.2
            if self._par.colormap is None:
                self._par.colormap = "seismic"
            if self._par.colorbarshrink is None:
                self._par.colorbarshrink = 0.4

        if kwargs:
            for key, val in kwargs.items():
                log.warning("Unknown argument '%s' with value '%s'.", key, str(val))

    def _pre_show(self):
        """Setup prior to plot"""
        if self._is_structured:
            self._data = self._ensemble["data"]
        else:
            self._data = self._ensemble

        if self._data.ndim == 1:
            self._data = np.reshape(self._data, (1, -1))
            if self._par.vaxis is None:
                if self._is_structured:
                    delay = self._ensemble[self._par.mnemonic_delrt]*1.e-3
                    dt = self._ensemble[self._par.mnemonic_dt]*1e-6
                else:
                    delay = 0
                    dt = 1
        elif self._data.ndim == 2:
            if self._par.vaxis is None:
                if self._is_structured:
                    delay = self._ensemble[self._par.mnemonic_delrt][0]*1.e-3
                    dt = self._ensemble[self._par.mnemonic_dt][0]*1e-6
                else:
                    delay = 0
                    dt = 1
        else:
            raise RuntimeError("Cannot display arrays of ndim > 2.")

        nt, ns = self._data.shape

        if self._par.vaxis is None:
            self._par.vaxis = np.arange(delay, delay+(ns-1)*dt+dt/2, dt)
        else:
            self._par.vaxis = np.atleast_1d(self._par.vaxis)  # make sure we have a numpy array
            dt = self._par.vaxis[1] - self._par.vaxis[0]

        if len(self._par.vaxis) != ns:
            raise ValueError(f"The size of 'vaxis' {len(self._par.vaxis)} "
                             f"does not match the data (ns={ns}).")

        if self._par.plottype != "spectrum":
            if self._par.haxis is None:
                self._par.haxis = np.arange(nt)
            elif isinstance(self._par.haxis, str):
                mnemonic = self._par.haxis
                if self._is_structured:
                    self._par.haxis = np.atleast_1d(self._ensemble[mnemonic])
                else:
                    self._par.haxis = np.arange(nt)
            else:
                self._par.haxis = np.atleast_1d(self._par.haxis)  # make sure we have a numpy array
            if len(self._par.haxis) != nt:
                raise ValueError(f"The size of 'haxis' {len(self._par.haxis)} "
                                 f"does not match the data (nt={nt}).")
        else:
            if self._par.nfft is None:
                self._par.nfft = ns
            if self._par.haxis is None:
                self._par.haxis = np.fft.rfftfreq(self._par.nfft, d=dt)
            else:
                raise ValueError("In 'spectrum' mode, paramter 'haxis' needs to be 'None'.")

        if self._par.fig is None and self._par.ax is None:
            self._setup_figure()

        self._par.ax.set_facecolor(self._par.facecolor)

        self._scale_data()

    def _post_show(self, axi):
        """Setup posterior to plot"""
        self._set_limits()
        self._set_ticks()
        self._set_grid()
        self._set_label()
        self._set_colorbar(axi)
        self._overlay()
        if self._par.tight:
            self.tight()

    def show(self):
        """
        Render and display a seismic plot.

        Returns
        -------
        figure.Figure, axes.Axes or figure.Figure, (axes.Axes, axes.Axes)
            Matplotlib's figure.Figure and axes.Axes object. If an overlay
            is used, the twinx-axis is returned, too.
        """
        self._pre_show()

        axi = None
        if self._par.plottype == "image":
            axi = self._image()
        elif self._par.plottype == "wiggle":
            self._wiggle()

        self._post_show(axi)

        if self._par.file is not None:
            self._par.fig.savefig(self._par.file, dpi=self._par.dpi, bbox_inches='tight')

        if self._par.ax_ovl is not None:
            return self._par.fig, (self._par.ax, self._par.ax_ovl)
        else:
            return self._par.fig, self._par.ax

    def _setup_figure(self):
        """Create a new figure if necessary."""
        plt.style.use(self._par.style)
        self._par.fig, self._par.ax = \
            plt.subplots(1, 1, figsize=(self._par.width, self._par.height))

    def _scale_data(self):
        """Scale the data according to user-requested normalization."""
        if self._par.normalize == "trace":
            scalers = np.abs(self._data).max(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                self._data = np.true_divide(self._data, scalers[:, None])
            self._data[~np.isfinite(self._data)] = 0  # -inf inf NaN
        elif self._par.normalize == "section":
            self._data = np.true_divide(self._data, np.abs(self._data).max())
        else:
            pass

    def _percentile(self):
        """Determine the percentile of the data."""
        self._par.vmm = np.percentile(np.fabs(self._data), self._par.perc)
        if self._par.vmm == 0:
            self._par.vmm = np.percentile(np.fabs(self._data), 100)

    def _clip(self):
        """Determine clip values for plotting."""
        if self._par.lowclip is not None:
            vmin = self._par.lowclip
        else:
            vmin = -self._par.vmm
        if self._par.highclip is not None:
            vmax = self._par.highclip
        else:
            vmax = self._par.vmm
        return (vmin, vmax)

    def _image(self, animated=False):
        """Create an image plot of the data."""
        if not animated:
            self._percentile()
        vmin, vmax = self._clip()
        norm = cm.colors.Normalize(vmax=vmax, vmin=vmin)
        axi = self._par.ax.imshow(self._data.T, cmap=self._par.colormap, norm=norm,
                                  interpolation=self._par.interpolation,
                                  alpha=self._par.alpha, origin="upper", aspect="auto",
                                  extent=[self._par.haxis[0], self._par.haxis[-1],
                                          self._par.vaxis.max(), self._par.vaxis.min()],
                                  animated=animated)
        return axi

    def _wiggle(self):
        """Create a wiggle plot of the data."""
        hpos = np.array(self._par.haxis[::self._par.skip])
        dataplt = self._data[::self._par.skip, :]
        ns = dataplt.shape[1]

        if len(hpos) > 1:
            spacing = np.min(np.abs(np.diff(hpos)))
        else:
            spacing = 1

        if self._par.haxisbeg is None:
            self._par.haxisbeg = hpos[0]-0.99*spacing
        if self._par.haxisend is None:
            self._par.haxisend = hpos[-1]+0.99*spacing

        scale = np.percentile(np.fabs(dataplt), self._par.perc)
        if scale == 0:
            scale = np.percentile(np.fabs(dataplt), 100)
        scale *= self._par.ampfac

        label = "_dummy"
        if self._par.label is not None:
            label = self._par.label

        for trace, curhpos in zip(dataplt, hpos):
            amp = trace / scale * spacing + curhpos
            if self._par.wigglehires:
                itpl_vaxis = np.linspace(self._par.vaxis[0], self._par.vaxis[-1], 10*ns)
                itpl_amp = np.interp(itpl_vaxis, self._par.vaxis, amp)
            else:
                itpl_vaxis = self._par.vaxis
                itpl_amp = amp

            # clip based on xcur parameter
            clipmin = curhpos-self._par.xcur*spacing
            clipmax = curhpos+self._par.xcur*spacing
            if clipmax < clipmin:
                clipmin, clipmax = clipmax, clipmin

            np.clip(itpl_amp, clipmin, clipmax, out=itpl_amp)

            if self._par.wiggledraw:
                self._par.ax.plot(itpl_amp, itpl_vaxis, self._par.linecolor,
                                  lw=self._par.linewidth, label=label)
            if self._par.wigglefill:
                if self._par.fillneg:
                    self._par.ax.fill_betweenx(itpl_vaxis, itpl_amp, curhpos, lw=0,
                                               where=itpl_amp < curhpos, alpha=self._par.alpha,
                                               facecolor=self._par.fillcolor)
                else:
                    self._par.ax.fill_betweenx(itpl_vaxis, itpl_amp, curhpos, lw=0,
                                               where=itpl_amp > curhpos, alpha=self._par.alpha,
                                               facecolor=self._par.fillcolor)
            # reset so we don't end up with ntraces legend entries
            label = "_dummy"

        self._par.ax.invert_yaxis()
        self._par.ax.set_ylim([self._par.vaxis.max(), self._par.vaxis.min()])

    def _set_limits(self):
        """Set the x- and y-limits of the plot."""
        self._par.ax.set_xlim(left=self._par.haxisbeg, right=self._par.haxisend)
        self._par.ax.set_ylim(top=self._par.vaxisbeg, bottom=self._par.vaxisend)

    def _set_ticks(self):
        """Make the necessary adjustments for all ticks."""
        if self._par.hmajorticks is not None:
            self._par.ax.xaxis.set_major_locator(MultipleLocator(self._par.hmajorticks))

        if self._par.hminorticks is not None:
            self._par.ax.xaxis.set_minor_locator(MultipleLocator(self._par.hminorticks))

        if self._par.vmajorticks is not None:
            self._par.ax.yaxis.set_major_locator(MultipleLocator(self._par.vmajorticks))

        if self._par.vminorticks is not None:
            self._par.ax.yaxis.set_minor_locator(MultipleLocator(self._par.vminorticks))

        if self._par.htickformat is None:
            if _is_integer(self._par.haxis):
                self._par.ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            # else:
            #     self._par.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            self._par.ax.xaxis.set_major_formatter(FormatStrFormatter(f"{self._par.htickformat}"))

        if self._par.vtickformat is None:
            if _is_integer(self._par.vaxis):
                self._par.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            # else:
            #     self._par.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            self._par.ax.yaxis.set_major_formatter(FormatStrFormatter(f"{self._par.vtickformat}"))

        if self._par.ticktop:
            self._par.ax.xaxis.tick_top()
            self._par.ax.xaxis.set_label_position('top')

        self._par.ax.tick_params(which='major', direction=self._par.tickdirection,
                                 labelsize=self._par.ticklabelsize,
                                 length=self._par.majorticklength,
                                 labelcolor=self._par.labelcolor,
                                 color=self._par.labelcolor,
                                 width=self._par.majortickwidth)

        self._par.ax.tick_params(which='minor', direction=self._par.tickdirection,
                                 length=self._par.minorticklength,
                                 labelcolor=self._par.labelcolor,
                                 color=self._par.labelcolor,
                                 width=self._par.minortickwidth)

        if self._par.hticklabelrot is not None:
            self._par.ax.tick_params(axis='x', labelrotation=self._par.hticklabelrot)
        if self._par.vticklabelrot is not None:
            self._par.ax.tick_params(axis='y', labelrotation=self._par.vticklabelrot)

    def _set_grid(self):
        """Make the necessary adjustments for plotting grids."""
        if self._par.hgrid:
            self._par.ax.grid(visible=True, axis="x", which=self._par.hgrid,
                              linestyle=self._par.gridstyle,
                              color=self._par.gridcolor,
                              alpha=self._par.gridlinealpha,
                              linewidth=self._par.gridlinewidth)
        else:
            self._par.ax.grid(visible=False, axis="x")
        if self._par.vgrid:
            self._par.ax.grid(visible=True, axis="y", which=self._par.vgrid,
                              linestyle=self._par.gridstyle,
                              color=self._par.gridcolor,
                              alpha=self._par.gridlinealpha,
                              linewidth=self._par.gridlinewidth)
        else:
            self._par.ax.grid(visible=False, axis="y")

    def _set_label(self):
        """Set axis labels and the title."""
        if self._par.hlabel is not None:
            self._par.ax.set_xlabel(self._par.hlabel,
                                    fontsize=self._par.labelfontsize,
                                    color=self._par.labelcolor,
                                    loc=self._par.hlabelpos)
        if self._par.vlabel is not None:
            self._par.ax.set_ylabel(self._par.vlabel,
                                    fontsize=self._par.labelfontsize,
                                    color=self._par.labelcolor,
                                    loc=self._par.vlabelpos)
        if self._par.title is not None:
            self._par.ax.set_title(self._par.title,
                                   fontsize=self._par.titlefontsize,
                                   color=self._par.titlecolor,
                                   loc=self._par.titlepos, fontweight='bold')

    def _set_colorbar(self, axi):
        """Set the colorbar."""
        if self._par.plottype  in ["wiggle", "spectrum"] or axi is None:
            return
        if self._par.colorbar:
            cbar = plt.colorbar(axi, ax=self._par.ax, orientation="vertical",
                                fraction=self._par.colorbarfraction,
                                shrink=self._par.colorbarshrink,
                                pad=self._par.colorbarpad)
            if self._par.colorbarbins is not None:
                cbar.ax.locator_params(nbins=self._par.colorbarbins)
        if self._par.colorbar and self._par.colorbarlabel:
            cbar.ax.set_ylabel(self._par.colorbarlabel,
                               fontsize=self._par.colorbarlabelsize,
                               color=self._par.labelcolor)
            cbar.ax.tick_params(which='major', direction="in",
                                labelsize=self._par.colorbarticklabelsize,
                                labelcolor=self._par.labelcolor,
                                color=self._par.labelcolor)
            cbar.ax.get_yaxis().labelpad = self._par.colorbarlabelpad

    def _overlay(self):
        """Plot header overlay."""
        if self._par.overlay is None:
            return

        if self._par.overlay == "rms":
            val = np.sqrt(np.mean(self._data**2, axis=-1, keepdims=True))
            val[val == 0] = np.finfo(np.float32).eps
        else:
            if not self._is_structured:
                log.warning("Data not supplied as Numpy structured array, cannot overlay header mnemonic '%s'.",
                            self._par.overlay)
                return
            val = np.atleast_1d(self._ensemble[self._par.overlay])

        if len(val) != len(self._par.haxis):
            log.warning("Length of overlay vector differs from length of horizontal axis - cancelling overlay.")
            return

        self._par.ax_ovl = self._par.ax.twinx()
        self._par.ax_ovl.plot(self._par.haxis, val, color=self._par.overlaycolor,
                              linewidth=self._par.overlaylinewidth)
        if self._par.ovlaxisbeg is None:
            self._par.ovlaxisbeg = val.min()
        if self._par.ovlaxisend is None:
            self._par.ovlaxisend = val.max()
        self._par.ax_ovl.set_ylim(self._par.ovlaxisbeg, self._par.ovlaxisend)
        if self._par.overlaylabel is not None:
            self._par.ax_ovl.set_ylabel(self._par.overlaylabel,
                                        fontsize=self._par.labelfontsize,
                                        color=self._par.overlaycolor,
                                        loc=self._par.vlabelpos)
        self._par.ax_ovl.yaxis.label.set_color(self._par.overlaycolor)
        self._par.ax_ovl.grid(visible=False, axis='both')
        self._par.ax_ovl.tick_params(which='major', direction=self._par.tickdirection,
                                     labelsize=self._par.ticklabelsize,
                                     length=self._par.majorticklength,
                                     labelcolor=self._par.overlaycolor,
                                     color=self._par.overlaycolor,
                                     width=self._par.majortickwidth)
        self._par.ax_ovl.tick_params(which='minor', direction=self._par.tickdirection,
                                     length=self._par.minorticklength,
                                     labelcolor=self._par.overlaycolor,
                                     color=self._par.overlaycolor,
                                     width=self._par.minortickwidth)
        if self._par.vticklabelrot is not None:
            self._par.ax_ovl.tick_params(axis='y', labelrotation=self._par.vticklabelrot)
        if self._par.ovlmajorticks is not None:
            self._par.ax_ovl.yaxis.set_major_locator(MultipleLocator(self._par.ovlmajorticks))
        if self._par.ovlminorticks is not None:
            self._par.ax_ovl.yaxis.set_minor_locator(MultipleLocator(self._par.ovlminorticks))
        self._par.ax_ovl.yaxis.set_inverted(self._par.overlayinvert)

    def _toggle(self, alldata, interval=None, repeat_delay=None, blit=False):
        """Toggle seismic plots."""
        self._pre_show()
        axi = self._image(animated=False)
        self._post_show(axi)
        shp = self._data.shape

        ims = []
        ims.append([self._image(animated=True)])
        for i in range(1, len(alldata)):
            if alldata[i].dtype.names is not None:
                self._data = alldata[i]["data"]
            else:
                self._data = alldata[i]
            if self._data.shape != shp:
                raise ValueError("Data sets to animate differ in shape "
                                 f"({shp} vs. {self._data.shape}).")
            self._scale_data()
            ims.append([self._image(animated=True)])
        # reset data to original
        if self._is_structured:
            self._data = self._ensemble["data"]
        else:
            self._data = self._ensemble

        ani = animation.ArtistAnimation(self._par.fig, ims, interval=interval,
                                        blit=blit, repeat_delay=repeat_delay)
        # note: blit=True causes axes or ticks to disappear which seems to be
        #       a known issue also experienced by others

        return ani, self._par.fig, self._par.ax

    def _wipe(self, data1, data2, nwipe=None, direction=None, interval=None,
              repeat_delay=None, blit=False, drawwipe=True, wipecolor="black"):
        self._pre_show()
        axi = self._image(animated=False)
        self._post_show(axi)
        shp = self._data.shape

        if data2.dtype.names is not None:
            dd2 = data2["data"]
        else:
            dd2 = data2

        idx2 = []
        if direction == "lr":
            nn = shp[0]
            myaxis = self._par.haxis
            obeg = self._par.vaxis[0]
            oend = self._par.vaxis[-1]
            if drawwipe:
                line = self._par.ax.plot([myaxis[0], myaxis[0]], [obeg, oend],
                                         color=wipecolor, animated=True)
        else:
            nn = shp[1]
            myaxis = self._par.vaxis
            obeg = self._par.haxis[0]
            oend = self._par.haxis[-1]
            if drawwipe:
                line = self._par.ax.plot([obeg, oend], [myaxis[0], myaxis[0]],
                                         color=wipecolor, animated=True)
        # determine block sizes and make sure they are as equally distributed
        # as possible rather than simply making the first or last block larger
        steps_avg = nn // nwipe
        rem = nn % nwipe
        end = 0
        idx2.append((0, 0))
        for i in range(nwipe):
            if i < rem:
                end += steps_avg+1
            else:
                end += steps_avg
            idx2.append((0, end))

        ims = []
        if drawwipe:
            ims.append([self._image(animated=True), line[0]])
        else:
            ims.append([self._image(animated=True)])

        for i in range(1, len(idx2)):
            idx = idx2[i][1]
            if idx >= len(myaxis):
                idx -= 1
            ll = myaxis[idx]
            if direction == "lr":
                self._data[idx2[i][0]:idx2[i][1], :] = dd2[idx2[i][0]:idx2[i][1], :]
                if drawwipe:
                    line = self._par.ax.plot([ll, ll], [obeg, oend],
                                             color=wipecolor, animated=True)
            else:
                self._data[:, idx2[i][0]:idx2[i][1]] = dd2[:, idx2[i][0]:idx2[i][1]]
                if drawwipe:
                    line = self._par.ax.plot([obeg, oend], [ll, ll],
                                             color=wipecolor, animated=True)
            self._scale_data()
            if drawwipe:
                ims.append([self._image(animated=True), line[0]])
            else:
                ims.append([self._image(animated=True)])
        # add reverse direction
        i = len(ims)-2
        while i >= 0:
            ims.append(ims[i])
            i -= 1

        # reset data to original
        if self._is_structured:
            self._data = self._ensemble["data"]
        else:
            self._data = self._ensemble

        ani = animation.ArtistAnimation(self._par.fig, ims, interval=interval,
                                        blit=blit, repeat_delay=repeat_delay)
        # note: blit=True causes axes or ticks to disappear which seems to be
        #       a known issue also experienced by others

        return ani, self._par.fig, self._par.ax

    def _spectrum(self):
        self._pre_show()
        nt, ns = self._data.shape
        dt = (self._par.vaxis[1] - self._par.vaxis[0]).astype(np.float64)
        df = (self._par.haxis[1] - self._par.haxis[0]).astype(np.float64)

        label = "_dummy"
        if self._par.label is not None:
            label = self._par.label

        if self._par.window is not None:
            winbuf = self._par.window(ns).astype(np.float64)
        else:
            winbuf = np.ones(ns, dtype=np.float32)
        spec = dt*np.fft.rfft(self._data.astype(np.float64)*winbuf,
                              n=self._par.nfft, norm=self._par.fftnorm)

        if self._par.haxisbeg is None:
            self._par.haxisbeg = self._par.haxis[0]
        if self._par.haxisend is None:
            self._par.haxisend = self._par.haxis[-1]

        mask = np.argwhere(np.logical_and(self._par.haxis >= self._par.haxisbeg,
                                          self._par.haxis <= self._par.haxisend))

        amp_ = np.abs(spec)
        pha_ = np.angle(spec)
        freq_ = self._par.haxis[mask]
        del spec

        # When A = rfft(a) and fs is the sampling frequency, A[0] contains the
        # zero-frequency term 0*fs, which is real due to Hermitian symmetry.
        # If n is even, A[-1] contains the term representing both positive and
        # negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
        # real. If n is odd, there is no term at fs/2; A[-1] contains the largest
        # positive frequency (fs/2*(n-1)/n), and is complex in the general case.

        if self._par.ampspec:
            del pha_
            # average over all traces
            amp = np.mean(amp_, axis=0)
            if self._par.smooth:
                if self._par.smoothwindow is None:
                    self._par.smoothwindow = 5*df
                nwin = int(self._par.smoothwindow/df)
                if nwin < 1:
                    nwin = 1
                ampspec = _mva(amp, nwin)
            else:
                ampspec = amp
            if self._par.scale != "linear":
                # ampspec[ampspec <= 0.0] = np.finfo(np.float64).eps
                with np.errstate(all="ignore"):
                    ampspec = 20.*np.log10(ampspec/np.max(ampspec))
            # mask array to get proper default axis ranges rather than just cutting axes afterwards
            self._par.ax.plot(freq_, ampspec[mask], color=self._par.linecolor,
                              lw=self._par.linewidth, label=label)
        else:
            del amp_
            # DC and Nyquist component have no imaginary part as input is real;
            # explicitly setting phase to zero helps unwrapping later on
            pha_[:, 0] = 0.0
            if np.mod(self._par.nfft, 2) == 0:
                pha_[:, -1] = 0.0
            # average over all traces
            pha = np.mean(pha_, axis=0)
            if self._par.unwrap:
                pha = np.unwrap(pha)
            if self._par.degree:
                pha = np.rad2deg(pha)
            # mask array to get proper default axis ranges rather than just cutting axes afterwards
            self._par.ax.plot(freq_, pha[mask], color=self._par.linecolor,
                              lw=self._par.linewidth, label=label)

        self._post_show(None)
        if self._par.file is not None:
            self._par.fig.savefig(self._par.file, dpi=self._par.dpi, bbox_inches='tight')

        return self._par.fig, self._par.ax

    def _crossplot(self):
        headers = None
        if self._is_structured:
            headers = self._ensemble.dtype.names
            if "data" in headers:
                self._data = self._ensemble["data"].astype(np.float32)
            else:
                self._data = None
        else:
            import pandas as pd
            if not isinstance(self._ensemble, pd.DataFrame):
                raise RuntimeError("Crossplots need a Numpy structured array (or Pandas dataframe) as input data.")
            else:
                self._data = None
                headers = self._ensemble.columns.tolist()

        if self._par.fig is None and self._par.ax is None:
            self._setup_figure()

        self._par.ax.set_facecolor(self._par.facecolor)

        if self._par.size_by is not None:
            if self._par.size_by not in headers:
                raise ValueError(f"Specified 'size_by={self._par.size_by}' is not a valid header mnemonic.")
            size = self._ensemble[self._par.size_by] * self._par.size_scale
        else:
            size = self._par.markersize

        if self._par.equal_axes:
            self._par.ax.axes.set_aspect('equal')

        if self._par.hkey not in headers:
            raise ValueError(f"Specified 'hkey={self._par.hkey}' is not a valid header mnemonic.")
        if self._par.vkey not in headers:
            raise ValueError(f"Specified 'vkey={self._par.vkey}' is not a valid header mnemonic.")
        self._par.haxis = self._ensemble[self._par.hkey] * self._par.hscale
        self._par.vaxis = self._ensemble[self._par.vkey] * self._par.vscale

        if isinstance(self._par.color_by, str):
            if self._par.color_by.lower() == "header":
                if self._par.ckey not in headers:
                    raise ValueError(f"Specified 'ckey={self._par.ckey}' is not a valid header mnemonic.")
                color_by = self._ensemble[self._par.ckey]
            elif self._par.color_by.lower() == "array":
                if self._par.ckey is None or not isinstance(self._par.ckey, np.ndarray):
                    raise RuntimeError("Need a Numpy array as 'ckey' to color the requested plot.")
                else:
                    color_by = self._par.ckey
            else:
                if self._data is None:
                    raise RuntimeError("Need a structured array with data values to color the requested plot.")
            if self._par.color_by.lower() == "rms":
                color_by = np.sqrt(np.mean(self._data**2, axis=-1))
            elif self._par.color_by.lower() == "mean":
                color_by = np.mean(self._data, axis=-1)
            elif self._par.color_by.lower() == "median":
                color_by = np.median(self._data, axis=-1)
            elif self._par.color_by.lower() == "min":
                color_by = np.min(self._data, axis=-1)
            elif self._par.color_by.lower() == "max":
                color_by = np.max(self._data, axis=-1)
            elif self._par.color_by.lower() == "fpeak":
                nt, ns = self._data.shape
                dt = self._ensemble[self._par.mnemonic_dt][0] * 1e-6
                freq = np.fft.rfftfreq(n=ns, d=dt).astype(np.float32)
                df = freq[1] - freq[0]
                if self._par.window is not None:
                    winbuf = self._par.window(ns).astype(np.float32)
                else:
                    winbuf = np.ones(ns, dtype=np.float32)
                amp = np.abs(np.fft.rfft(self._data*winbuf, n=ns, axis=-1))
                if self._par.smooth:
                    if self._par.smoothwindow is None:
                        self._par.smoothwindow = 5*df
                    nwin = int(self._par.smoothwindow/df)
                    if nwin < 1:
                        nwin = 1
                    ampspec = _mva(amp, nwin)
                else:
                    ampspec = amp
                idx = np.argmax(ampspec, axis=-1).copy()
                color_by = freq[idx].copy()
                del ampspec, amp, freq, winbuf
            if self._par.lowclip is None:
                self._par.lowclip = np.min(color_by)
            if self._par.highclip is None:
                self._par.highclip = np.max(color_by)

        axi = self._par.ax.scatter(x=self._par.haxis, y=self._par.vaxis,
                                   s=size, marker=self._par.marker, c=color_by,
                                   alpha=self._par.alpha, cmap=self._par.colormap,
                                   edgecolors=self._par.edgecolor,
                                   norm=self._par.color_norm, vmin=self._par.lowclip,
                                   vmax=self._par.highclip, linewidth=self._par.linewidth,
                                   label=self._par.label, figure=self._par.fig)
        self._par.ax.yaxis.set_inverted(self._par.vinvert)
        self._par.ax.xaxis.set_inverted(self._par.hinvert)

        ax_hist = None
        if self._par.histogram and isinstance(self._par.color_by, str):
            ax_hist = self._par.ax.inset_axes([0, self._par.histogram_vpos,
                                               1, self._par.histogram_height])
            counts, bins = np.histogram(color_by, bins=self._par.histogram_bins,
                                        density=self._par.histogram_density)
            if self._par.histogram_colorize:
                cmap = plt.get_cmap(name=self._par.colormap)
                norm = Normalize(vmin=self._par.lowclip, vmax=self._par.highclip)
                c = cmap(norm(bins))
            else:
                c = self._par.histogram_color
            ax_hist.bar(bins[:-1], counts, width=np.diff(bins), color=c)
            ax_hist.set_xlim(self._par.lowclip, self._par.highclip)
            if self._par.histogram_vlabel is not None:
                if self._par.histogram_vlabel.lower() == "auto":
                    if self._par.histogram_density:
                        ax_hist.set_ylabel("density")
                    else:
                        ax_hist.set_ylabel("count")
                else:
                    ax_hist.set_ylabel(self._par.histogram_vlabel)
            if self._par.histogram_hlabel is not None:
                if self._par.histogram_hlabel.lower() == "auto":
                    if self._par.color_by.lower() == "header":
                        ax_hist.set_xlabel(f"{self._par.ckey}")
                    elif self._par.color_by.lower() == "rms":
                        ax_hist.set_xlabel("RMS amplitude")
                    elif self._par.color_by.lower() == "mean":
                        ax_hist.set_xlabel("mean amplitude")
                    elif self._par.color_by.lower() == "median":
                        ax_hist.set_xlabel("median amplitude")
                    elif self._par.color_by.lower() == "min":
                        ax_hist.set_xlabel("minimum amplitude")
                    elif self._par.color_by.lower() == "max":
                        ax_hist.set_xlabel("maximum amplitude")
                    elif self._par.color_by.lower() == "fpeak":
                        ax_hist.set_xlabel("peak frequency")
                else:
                    ax_hist.set_xlabel(self._par.histogram_hlabel)

        self._post_show(axi)
        if self._par.file is not None:
            self._par.fig.savefig(self._par.file, dpi=self._par.dpi,
                                  bbox_inches='tight')

        if ax_hist is not None:
            return self._par.fig, (self._par.ax, ax_hist)
        else:
            return self._par.fig, self._par.ax


@jit(nopython=True)
def _mva(data, filter_size):
    indexer = filter_size // 2
    temp = np.zeros((filter_size, ), dtype=data.dtype)
    data_final = np.zeros_like(data)
    nrow = len(data)
    for j in np.arange(nrow):
        for z in np.arange(filter_size):
            if j+z-indexer < 0:
                temp[z] = data[0]
            elif j+indexer > nrow-1:
                temp[z] = data[-1]
            else:
                temp[z] = data[j+z-indexer]
        data_final[j] = temp.mean()
    return data_final
