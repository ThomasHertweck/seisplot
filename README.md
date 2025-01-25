# seisplot

Plotting of seismic data using variable-density or variable-area wiggle displays.

## Description

The seisplot module provides a single, highly flexible method *plot()* to
display seismic data. A second method *toggle()* can be used to create
animations that can be saved as images or movies.

The module was designed to get decent displays of seismic data in a way that
is more convenient than going through many individual Matplotlib function calls
explicitly. However, if required Matplotlib methods can also be called directly.
The code is pure Python and kept deliberately simple to get students
participating our Geophysics classes and exercises going with Python and
seismic data.

## Key features

* Variable-density image plots.
* Variable-area wiggle plots.
* Highly configurable settings like colors, line widths, colorbars, labels,
  axis ticks, grid lines, etc.
* Animated toggles between two or more seismic image plots.
* Animated wipes between two seismic image plots.

<p align="center">
![Image plot](img/img1.png "Variable-density image plot")
![Wiggle plot](img/img2.png "Variable-area wiggle plot")
![Velocity plot](img/img3.png "Non-seismic data plot")
![Fielddata plot](img/img4.png "Trace-normalized field data plot")
![Toggle plot](img/img5.apng "Toggle of image plots")
![Wipe plot](img/img6.apng "Wipe of image plots")

</p>

## Getting Started

### Dependencies

Required: numpy, matplotlib

### Installation

*Install from PyPI:*

```
$> pip install seisplot
```

*Install directly from gitlab:*

```
$> pip install git+https://gitlab.kit.edu/thomas.hertweck/seisplot.git
```

*Editable install from source:*

This version is intended for experts who would like to test the latest
version or make modifications. Normal users should prefer to install a
stable version.

```
$> git clone https://gitlab.kit.edu/thomas.hertweck/seisplot.git
```

Once you acquired the source, you can install an editable version of seisplot
with:

```
$> cd seisplot
$> pip install -e .
```

## Brief tutorial

For a demonstration of various features and much more, please visit the
"examples" folder where several Jupyter notebooks (tutorials) are available.

Plotting seismic data (for instance, read with our __seisio__ package and
therefore available as Numpy structured array including trace headers) can be
as simple as:

```
import seisplot

fig, ax = seisplot.plot(data, haxis="offset", width=4, height=6,
                        vlabel="Time (s)", hlabel="Offset (m)",
                        vmajorticks=0.2, vminorticks=0.1,
                        hminorticks=500, vgrid="major")
```
The variables `fig` and `ax` are standard Matplotlib figure and axis handles
that can be used to tweak the display further. You could also create those
first using `fig, ax = plt.subplots(1, 1)` and pass them to the *plot()* method.
In this way, it is possible to, for instance, create several seismic displays
in one figure, or create displays that share the y-axis (usually "time").

A display toggle can, for instance, be produced in the following way:

```
ani, fig, ax = seisplot.toggle([data_1, data_2, data_diff],
                               interval=1000, repeat_delay=0,
                               hlabel="offset (m)", vlabel="time (s)")
```
The returned animation-artist object can be used to save an animated image
or a movie.

An animated wipe can, for instance, be produced in the following way:

```
ani, fig, ax = seisplot.wipe(data_1, data_2, blit=True,
                             nwipe=30, wipecolor="blue",
                             interval=5, repeat_delay=0,
                             hlabel="offset (m)", vlabel="time (s)")
```
Again, the returned animation-artist object can be used to save an animated
image or a movie.

## Main author

Dr. Thomas Hertweck, geophysics@email.de

## License

This project is licensed under the LGPL v3.0 License - see the LICENSE.md
file for details
