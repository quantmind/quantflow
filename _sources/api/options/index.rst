==================================
Option Pricing
==================================

.. currentmodule:: quantflow.options

The :mod:`options` module provides classes and functions for pricing and analyzing options.
The main class is the :class:`.VolSurface` class which is used to represent
a volatility surface for a given asset.
A volatility surface is usually created via a :class:`.VolSurfaceLoader` object.

.. toctree::
    :maxdepth: 1

    vol_surface
    black
    pricer
    calibration
