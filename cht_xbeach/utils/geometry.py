"""Geometry primitives and grid helpers used by cht_xbeach.

Provides ``RegularGrid``, ``Point``, and ``Polyline`` classes for constructing,
describing, and exporting XBeach-compatible computational grids.
"""

import math
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely


class Geometry:
    """Abstract base for geometry objects."""

    def __init__(self) -> None:
        pass


class RegularGrid(Geometry):
    """Axis-aligned or rotated regular structured grid.

    Parameters
    ----------
    hw : any
        Parent object reference (not used internally but kept for API
        compatibility).
    x0 : float, optional
        X coordinate of the grid origin.
    y0 : float, optional
        Y coordinate of the grid origin.
    dx : float, optional
        Cell size in the X direction.
    dy : float, optional
        Cell size in the Y direction.
    nmax : int, optional
        Number of cells in the Y (n) direction.
    mmax : int, optional
        Number of cells in the X (m) direction.
    rotation : float, optional
        Grid rotation angle in degrees (anti-clockwise from east).
    crs : any, optional
        Coordinate reference system descriptor.
    """

    def __init__(
        self,
        hw,
        x0: float = None,
        y0: float = None,
        dx: float = None,
        dy: float = None,
        nmax: int = None,
        mmax: int = None,
        rotation: float = None,
        crs=None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = mmax
        self.nmax = nmax
        self.rotation = rotation
        self.crs = crs
        if x0:
            self.xg, self.yg = self.grid_coordinates_corners()
            self.xz, self.yz = self.grid_coordinates_centres()

    def build(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int,
        ny: int,
        rotation: float,
        crs,
    ) -> None:
        """Populate all grid attributes and compute coordinate arrays.

        Parameters
        ----------
        x0 : float
            X coordinate of the grid origin.
        y0 : float
            Y coordinate of the grid origin.
        dx : float
            Cell size in the X direction.
        dy : float
            Cell size in the Y direction.
        nx : int
            Number of cells in the X direction.
        ny : int
            Number of cells in the Y direction.
        rotation : float
            Grid rotation angle in degrees.
        crs : any
            Coordinate reference system descriptor.
        """
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = nx
        self.nmax = ny
        self.rotation = rotation
        self.xg, self.yg = self.grid_coordinates_corners()
        self.xz, self.yz = self.grid_coordinates_centres()
        self.crs = crs

    def grid_coordinates_corners(self):
        """Compute X/Y arrays at grid-cell corners.

        Returns
        -------
        xg : numpy.ndarray
            X coordinates, shape ``(nmax+1, mmax+1)``.
        yg : numpy.ndarray
            Y coordinates, shape ``(nmax+1, mmax+1)``.
        """
        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(0.0, self.mmax * self.dx, num=self.mmax + 1)
        yy = np.linspace(0.0, self.nmax * self.dy, num=self.nmax + 1)
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.x0 + xg0 * cosrot - yg0 * sinrot
        yg = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xg, yg

    def grid_coordinates_centres(self):
        """Compute X/Y arrays at grid-cell centres.

        Returns
        -------
        xz : numpy.ndarray
            X coordinates, shape ``(nmax, mmax)``.
        yz : numpy.ndarray
            Y coordinates, shape ``(nmax, mmax)``.
        """
        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(
            0.5 * self.dx, self.mmax * self.dx - 0.5 * self.dx, num=self.mmax
        )
        yy = np.linspace(
            0.5 * self.dy, self.nmax * self.dy - 0.5 * self.dy, num=self.nmax
        )
        xg0, yg0 = np.meshgrid(xx, yy)
        xz = self.x0 + xg0 * cosrot - yg0 * sinrot
        yz = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xz, yz

    def plot(self, ax) -> None:
        """Placeholder for grid plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on (currently unused).
        """
        pass

    def to_gdf(self) -> gpd.GeoDataFrame:
        """Export grid lines as a GeoDataFrame.

        Returns
        -------
        gdf : geopandas.GeoDataFrame
            Single-row GeoDataFrame containing a ``MultiLineString`` of all
            grid edges.
        """
        lines = []

        cosrot = math.cos(self.rotation * math.pi / 180)
        sinrot = math.sin(self.rotation * math.pi / 180)

        for n in range(self.nmax):
            for m in range(self.mmax):
                xa = self.x0 + m * self.dx * cosrot - n * self.dy * sinrot
                ya = self.y0 + m * self.dx * sinrot + n * self.dy * cosrot
                xb = self.x0 + (m + 1) * self.dx * cosrot - n * self.dy * sinrot
                yb = self.y0 + (m + 1) * self.dx * sinrot + n * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
                xb = self.x0 + m * self.dx * cosrot - (n + 1) * self.dy * sinrot
                yb = self.y0 + m * self.dx * sinrot + (n + 1) * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
        geom = shapely.geometry.MultiLineString(lines)
        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=[geom])

        return gdf


class Point:
    """Simple 2-D point with optional name and CRS.

    Parameters
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    name : str, optional
        Human-readable label.
    crs : any, optional
        Coordinate reference system descriptor.
    """

    def __init__(
        self,
        x: float,
        y: float,
        name: Optional[str] = None,
        crs=None,
    ) -> None:
        self.x = x
        self.y = y
        self.crs = crs
        self.name = name
        self.data = None


class Polyline(Geometry):
    """Ordered collection of points forming an open or closed polyline.

    Parameters
    ----------
    x : array-like, optional
        X coordinates of the vertices.
    y : array-like, optional
        Y coordinates of the vertices.
    crs : any, optional
        Coordinate reference system descriptor.
    name : str, optional
        Label for this polyline.
    closed : bool, optional
        Whether the polyline is closed (i.e. forms a polygon).  Default is
        ``False``.
    """

    def __init__(
        self,
        x=None,
        y=None,
        crs=None,
        name: Optional[str] = None,
        closed: bool = False,
    ) -> None:
        self.point = []
        self.name = name
        self.data = None
        self.closed = closed
        self.crs = crs

        if x is not None:
            for j, xp in enumerate(x):
                pnt = Point(x[j], y[j])
                self.point.append(pnt)

    def add_point(
        self,
        x: float,
        y: float,
        name: Optional[str] = None,
        data=None,
        position: int = -1,
    ) -> None:
        """Append or insert a new point into the polyline.

        Parameters
        ----------
        x : float
            X coordinate of the new point.
        y : float
            Y coordinate of the new point.
        name : str, optional
            Label for the point.
        data : any, optional
            Arbitrary data to attach to the point.
        position : int, optional
            Insertion index.  ``-1`` (default) appends to the end.
        """
        pnt = Point(x, y, name=name, data=data)
        if position < 0:
            # Add point to the end
            self.point.append(pnt)
        else:
            pass

    def plot(self, ax=None) -> None:
        """Placeholder for polyline plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (currently unused).
        """
        pass
