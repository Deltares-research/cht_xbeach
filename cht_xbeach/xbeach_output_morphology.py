"""XBeach morphological output analysis: beach profiles, dune crests, and Sallenger regimes.

Provides ``Profile`` for 1-D cross-shore analysis and ``Map`` for 2-D
alongshore regime mapping using XBeach simulation output.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.markers import CARETDOWN
from scipy import signal


class Profile:
    """1-D beach profile with interpolation, smoothing, and regime classification.

    ``x`` and ``z`` should always be ordered from land to sea (left to right).

    Parameters
    ----------
    x : numpy.ndarray
        Cross-shore location in metres.
    z : numpy.ndarray
        Pre-storm elevation at each point in *x*, in metres relative to MSL.
    zbend : numpy.ndarray
        Post-storm elevation at each point in *x*, in metres relative to MSL.
    win : float, optional
        Window size in metres for smoothing (should be a multiple of the grid
        spacing).
    smoothing : {'hanning', 'moving_average'}, optional
        Smoothing method to apply when *win* is given.
    ref : {'MSL', 'MHW'}, optional
        Reference datum used to locate the shoreline.  Default is ``'MSL'``.
    norm : bool, optional
        When ``True`` the x-origin is shifted to the shoreline.  Default is
        ``False``.
    **kwargs
        ``wls`` : list of float
            ``[MLW, MHW]`` mean low and high water levels relative to the
            input bed-level datum.
        ``id`` : str or int
            Profile / transect identifier.
    """

    def __init__(
        self,
        x,
        z,
        zbend,
        win: Optional[float] = None,
        smoothing: Optional[str] = None,
        ref: str = "MSL",
        norm: bool = False,
        **kwargs,
    ) -> None:
        if "wls" in kwargs:
            wls = kwargs.get("wls")
            self.MHW = max(wls)
            self.MLW = min(wls)
        else:
            self.MHW = 0
            self.MLW = 0
        if "id" in kwargs:
            self.id = str(kwargs.get("id"))

        self.x = x
        self.z = z
        self.zend = zbend

        if win:  # if window for smoothing is provided apply smoothing (should be multiple of dx)
            if smoothing == "moving_average":
                self.z = mov_average(self.x, self.z, win)
            if smoothing == "hanning":
                self.z = hanning_filter(self.x, self.z, win)
        if ref == "MHW":
            x0, ind = zero_crossing(self.x, self.z - self.MHW, indices=True)
        else:
            x0, ind = zero_crossing(self.x, self.z, indices=True)
        if len(x0) == 0:
            x0 = self.x[-1]
            ind = self.x == x0
        else:
            x0, ind = x0[-1], ind[-1]  # get last one in case there are more
        if norm:
            self.offset = self.x[ind]
            self.x = self.x - self.offset
        self.x_shoreline = self.x[ind]  # get grid value (not interpolated)

    def dune_crest(
        self,
        zmin: float = 6,
        prom_l: float = 1,
        prom_r: float = 1,
        width: float = 1,
        fr_x_min: Optional[float] = None,
        fr_x_max: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Find the cross-shore location of the maximum and foredune crests.

        Parameters
        ----------
        zmin : float, optional
            Minimum elevation for foredune peaks.  Default is ``6``.
        prom_l : float, optional
            Required left prominence for foredune peak detection.  Default
            is ``1``.
        prom_r : float, optional
            Required right prominence for foredune peak detection.  Default
            is ``1``.
        width : float, optional
            Minimum peak width in metres.  Default is ``1``.
        fr_x_min : float, optional
            Minimum cross-shore distance of foredune from the shoreline.
        fr_x_max : float, optional
            Maximum cross-shore distance of foredune from the shoreline.
        """
        ind_shor = np.argwhere(self.x == self.x_shoreline)[0][
            0
        ]  # find index of shoreline

        # find location of maximum dune crest across the profile
        try:
            x_dc_max = self.x[np.argmax(self.z)]
            x_dc_max_post = self.x[np.argmax(self.zend)]
        except Exception:
            x_dc_max = np.nan
            x_dc_max_post = np.nan

        # find most seaward (1st dune row) fore-dune location (use height/width filtering)
        try:
            peaks, properties = signal.find_peaks(
                self.z[0:ind_shor], width=int(np.round(width / (self.x[1] - self.x[0])))
            )

            if fr_x_min is not None:
                peaks = peaks[
                    (self.x[peaks] < self.x_shoreline + fr_x_min)
                    & (self.x[peaks] > self.x_shoreline - fr_x_max)
                ]  # filter for cross-shore location

            maxpeakid = np.where(self.z[peaks] == np.amax(self.z[peaks]))
            self.x_dc_fr = self.x[
                peaks[maxpeakid]
            ]  # B) highest peak landward of shoreline (within fr_x_min)
        except Exception:
            self.x_dc_fr = self.x[0]  # np.nan

        # now same for post-storm profile: find most seaward (1st dune row) fore-dune location
        try:
            peakspost, properties = signal.find_peaks(
                self.zend[0:ind_shor],
                width=int(np.round(width / (self.x[1] - self.x[0]))),
            )

            if fr_x_min is not None:
                peakspost = peakspost[
                    (self.x[peakspost] < self.x_shoreline + fr_x_min)
                    & (self.x[peakspost] > self.x_shoreline - fr_x_max)
                ]  # filter for cross-shore location

            maxpeakidpost = np.where(
                self.zend[peakspost] == np.amax(self.zend[peakspost])
            )
            self.x_dc_fr_post = self.x[
                peakspost[maxpeakidpost]
            ]  # B) highest peak landward of shoreline (within fr_x_min)
        except Exception:
            self.x_dc_fr_post = self.x[0]  # np.nan

        self.z_dc_max = self.z[self.x == x_dc_max][0]
        self.z_dc_fr = self.z[self.x == self.x_dc_fr][0]
        self.x_dc_max = x_dc_max

        self.z_dc_fr_post = self.zend[self.x == self.x_dc_fr_post][0]
        self.x_dc_max_post = x_dc_max_post

    def dune_vol(self) -> None:
        """Compute the dune volume between dune crest and shoreline (m²/m).

        Currently a stub — not implemented.
        """
        pass

    def plot(self, ax=None, plot_show: bool = False, **kwargs) -> None:
        """Plot the pre- and post-storm bed levels and key morphological features.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.  A new figure is created if not provided.
        plot_show : bool, optional
            Call ``plt.show()`` after plotting.  Default is ``False``.
        **kwargs
            ``water_levels`` : array-like
                ``[MHW, MLW]`` water levels to shade.
            ``xlim`` : list of float
                X-axis limits.
            ``ylim`` : list of float
                Y-axis limits.
            ``plot_dir`` : str
                Directory in which to save the figure as
                ``profile_<id>.png``.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if hasattr(self, "id"):
            ax.set_title(f"transect id : {self.id}")
        ax.set_ylim([np.nanmin(self.z), np.nanmax(self.z) + 1])
        ax.set_xlim([np.nanmin(self.x), np.nanmax(self.x)])
        ax.set_ylabel("elevation [m to NAVD88]")
        ax.set_xlabel("cross-shore distance [m]")
        if "xlim" in kwargs:
            ax.set_xlim(kwargs.get("xlim"))
        if "ylim" in kwargs:
            ax.set_ylim(kwargs.get("ylim"))
        if "water_levels" in kwargs:
            ax.fill_between(
                self.x,
                np.repeat(self.MHW, len(self.z)),
                np.nanmin(self.z),
                color="#a8d7ed",
                zorder=-3,
            )
        ax.fill_between(
            self.x,
            self.z,
            np.nanmin(self.z),
            color="#e8cea9",
            zorder=-2,
            label="Pre-storm bed level",
        )
        ax.fill_between(
            self.x,
            self.zend,
            np.nanmin(self.zend),
            color="#c98c36",
            zorder=-1,
            alpha=0.5,
            label="Post-storm bed level",
        )
        ax.plot(self.x, self.zsmax, "b--", linewidth=0.8, zorder=0, label="max zs-max")
        ax.plot(self.x, self.zsmean, "b-", linewidth=0.8, zorder=0, label="max zs-mean")
        if hasattr(self, "x_dc_fr") and ~np.isnan(self.x_dc_fr):
            ax.scatter(
                self.x_dc_fr,
                self.z_dc_fr,
                s=25,
                marker=CARETDOWN,
                linewidths=0.2,
                facecolor="k",
                zorder=2,
                label="Foredune crest",
                clip_on=False,
            )

        if hasattr(self, "x_dt") and ~np.isnan(self.x_dt):
            ax.scatter(
                self.x_dt,
                self.z_dt,
                s=20,
                marker="o",
                edgecolors="k",
                linewidths=0.2,
                facecolor="r",
                zorder=2,
                label="Dune toe",
            )
        if hasattr(self, "x_shoreline") and ~np.isnan(self.x_shoreline):
            ax.scatter(
                self.x_shoreline,
                self.MHW,
                s=10,
                marker="s",
                edgecolors="k",
                linewidths=0.2,
                facecolor="r",
                zorder=2,
                label="Shoreline",
            )
        if hasattr(self, "regimeno"):
            ax.scatter(
                self.x_dc_fr,
                self.z_dc_fr_zbend,
                s=25,
                marker=CARETDOWN,
                linewidths=0.2,
                facecolor="#666666",
                zorder=2,
                label=self.regimena,
                clip_on=False,
            )
        if hasattr(self, "erosionregimeno"):
            ax.scatter(
                self.x_dc_fr_post,
                self.z_dc_fr_post,
                s=25,
                marker=CARETDOWN,
                linewidths=0.2,
                facecolor="#009900",
                zorder=2,
                label=self.erosionregimena,
                clip_on=False,
            )
        ax.legend(loc=0, prop={"size": 8})

        if plot_show:
            plt.show(block=False)
        if "plot_dir" in kwargs:
            plot_dir = kwargs.get("plot_dir")
            plt.savefig(
                os.path.join(plot_dir, f"profile_{self.id}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

    def sallenger_regimes(self, zsmean, zsmax, eps: float = 0.005):
        """Determine the Sallenger (2000) storm-impact regime for this profile.

        Parameters
        ----------
        zsmean : numpy.ndarray
            Time-mean water surface elevation from XBeach,
            shape ``(time, crossshore_nx)``.
        zsmax : numpy.ndarray
            Maximum water surface elevation from XBeach,
            shape ``(time, crossshore_nx)``.
        eps : float, optional
            Minimum flooding depth used in XBeach (default ``0.005`` m).

        Returns
        -------
        regimeno : int
            Regime number: 1 = collision, 2 = overwash, 3 = inundation,
            4 = breach.
        regimena : str
            Human-readable regime name.
        ind_crest : int
            Array index of the dune crest.
        """
        zsmean = np.nanmax(zsmean, axis=0)
        zsmax = np.nanmax(zsmax, axis=0)

        ind_crest = np.argwhere(self.x == self.x_dc_fr)[0][
            0
        ]  # find index of dune crest

        if zsmean[ind_crest] > self.MHW:  # collision
            regimeno = 1
            regimena = "collision"
        if zsmax[ind_crest] - eps > self.zend[ind_crest]:  # overwash
            regimeno = 2
            regimena = "overwash"
        if zsmean[ind_crest] - eps > self.zend[ind_crest]:  # inundation
            regimeno = 3
            regimena = "inundation"
        if self.zend[ind_crest] < self.MHW:  # breach
            regimeno = 4
            regimena = "breaching"

        self.z_dc_fr_zbend = self.zend[ind_crest]
        self.zsmax = zsmax
        self.zsmean = zsmean
        self.regimeno = regimeno
        self.regimena = regimena

        return regimeno, regimena, ind_crest

    def erosion_regimes(self):
        """Determine the erosion regime for this profile.

        Returns
        -------
        erosionregimeno : int
            Regime number: 1 = minor erosion, 2 = beach/dune face erosion,
            3 = dune retreat, 4 = breach.
        erosionregimena : str
            Human-readable regime name.
        maxeroid : int or None
            Index of the most landward point where erosion starts.
        """
        ind_shor = np.argwhere(self.x == self.x_shoreline)[0][
            0
        ]  # find index of shoreline
        ind_crest = np.argwhere(self.x == self.x_dc_fr)[0][
            0
        ]  # find index of dune crest pre-storm
        ind_crest_post = np.argwhere(self.x == self.x_dc_fr_post)[0][
            0
        ]  # find index of dune crest post-storm

        # find most landward point where we find erosion, starting from the shoreline point
        dz = self.z - self.zend
        maxeroid = next(
            (
                index
                for index, value in enumerate(reversed(dz[0:ind_shor]))
                if value <= 0
            ),
            None,
        )  # not 0 bed level change, but use a threshold of ~5cm
        maxeroid = ind_shor - 1 - maxeroid

        if maxeroid is None:
            erosionregimeno, erosionregimena = 1, "minor erosion"
        elif (ind_crest_post == ind_crest) and (
            self.z[ind_crest] == self.zend[ind_crest_post]
        ):
            erosionregimeno, erosionregimena = 2, "beach/dune face erosion"
        elif (ind_crest_post != ind_crest) and (self.zend[ind_crest] > self.MHW):
            erosionregimeno, erosionregimena = 3, "dune retreat"
        elif self.zend[ind_crest] < self.MHW:
            erosionregimeno, erosionregimena = 4, "breaching"

        self.erosionregimeno = erosionregimeno
        self.erosionregimena = erosionregimena
        self.maxeroid = maxeroid

        return erosionregimeno, erosionregimena, maxeroid


class Map:
    """2-D beach map from XBeach output.

    Parameters
    ----------
    x2D : numpy.ndarray
        2-D X coordinates of the XBeach grid.
    y2D : numpy.ndarray
        2-D Y coordinates of the XBeach grid.
    zb02D : numpy.ndarray
        Pre-storm bed levels on the 2-D grid.
    zbend2D : numpy.ndarray
        Post-storm bed levels on the 2-D grid.
    plot_dir : str
        Directory in which to save figures.
    """

    def __init__(self, x2D, y2D, zb02D, zbend2D, plot_dir: str) -> None:
        self.x = x2D
        self.y = y2D
        self.zend = zbend2D
        self.z0 = zb02D
        self.plot_dir = plot_dir

    def plot_bedlevelmap(self, **kwargs) -> None:
        """Plot pre-storm, post-storm, and sedimentation–erosion bed levels.

        Parameters
        ----------
        **kwargs
            ``axis`` : list
                Axis limits passed to ``plt.axis()``.
            ``plot_dir`` : str
                Directory in which to save the figure.
        """
        cmap = np.loadtxt(os.path.abspath("..//misc//GMT_globe.txt"))
        cmap = cmap / 255
        newcmp = ListedColormap(cmap, name="test")

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
        ax = gs.subplots(sharey=True)

        for ii in range(1, 4):
            if ii == 1:
                var = self.z0
                cmap = newcmp
                climv = 20
                cblabel = "bed level [m to NAVD88]"
                titlab = "pre-storm bed level"
            if ii == 2:
                var = self.zend
                cmap = newcmp
                climv = 20
                cblabel = "bed level [m to NAVD88]"
                titlab = "post-storm bed level"
            if ii == 3:
                var = self.zend - self.z0
                cmap = "RdBu"
                climv = 2
                cblabel = "sed-ero [m]"
                titlab = "sedimentation-erosion (post-pre)"

            plt.subplot(1, 3, ii)
            plt.pcolor(self.x, self.y, var, cmap=cmap, vmin=-climv, vmax=climv)
            plt.colorbar(label=cblabel, location="bottom")
            plt.contour(
                self.x, self.y, self.z0, levels=[0], colors="black", linewidths=0.3
            )
            plt.title(titlab)
            plt.axis("equal")
            if "axis" in kwargs:
                axis = kwargs.get("axis")
                plt.axis([axis])

        plt.show(block=False)
        if "plot_dir" in kwargs:
            plot_dir = kwargs.get("plot_dir")
            fig.savefig(
                os.path.join(plot_dir, "map_modeled_bedlevels"),
                dpi=500,
                bbox_inches="tight",
            )

    def plot_sallengerregimes(self, **kwargs) -> None:
        """Plot Sallenger regimes for each alongshore transect on the 2-D bathy map.

        Parameters
        ----------
        **kwargs
            ``plot_dir`` : str
                Directory in which to save the figure.
        """
        cmap = np.loadtxt(
            r"p:\11206085-onr-fhics\03_cosmos\configurations\GMT_globe.txt"
        )
        cmap = cmap / 255
        newcmp = ListedColormap(cmap, name="test")

        figure, ax = plt.subplots(1, 1)

        plt.pcolor(self.x, self.y, self.zend, cmap=newcmp, vmin=-20, vmax=20)
        plt.colorbar(label="bed level [m to NAVD88]")

        df = pd.DataFrame(
            {
                "x": self.x_crest,
                "y": self.y_crest,
                "regime": self.regimenos,
                "regimename": self.regimename,
            }
        )
        groups = df.groupby("regimename")
        regimecolors = {
            "collision": "green",
            "overwash": "yellow",
            "inundation": "orange",
            "breaching": "red",
        }
        for reg, dffs in groups:
            plt.scatter(
                dffs.x,
                dffs.y,
                s=3,
                label=reg,
                facecolor=regimecolors[reg],
                edgecolors="k",
                linewidths=0.1,
            )
        plt.legend(loc=0, title="Sallenger regime")

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        plt.axis("equal")
        plt.ylim([df["y"].min(), df["y"].max()])
        plt.show(block=False)

        if "plot_dir" in kwargs:
            plot_dir = kwargs.get("plot_dir")
            figure.savefig(
                os.path.join(plot_dir, "map_regimes"), dpi=500, bbox_inches="tight"
            )

    def plot_erosionregimes(self, **kwargs) -> None:
        """Plot erosion regimes for each alongshore transect on the 2-D bathy map.

        Parameters
        ----------
        **kwargs
            ``plot_dir`` : str
                Directory in which to save the figure.
        """
        cmap = np.loadtxt(
            r"p:\11206085-onr-fhics\03_cosmos\configurations\GMT_globe.txt"
        )
        cmap = cmap / 255
        newcmp = ListedColormap(cmap, name="test")

        figure, ax = plt.subplots(1, 1)

        plt.pcolor(self.x, self.y, self.zend, cmap=newcmp, vmin=-20, vmax=20)
        plt.colorbar(label="bed level [m to NAVD88]")

        df = pd.DataFrame(
            {
                "x": self.x_crest,
                "y": self.y_crest,
                "regime": self.erosionregimenos,
                "regimename": self.erosionregimename,
            }
        )
        groups = df.groupby("regimename")
        regimecolors = {
            "minor erosion": "green",
            "beach/dune face erosion": "yellow",
            "dune retreat": "orange",
            "breaching": "red",
        }
        for reg, dffs in groups:
            plt.scatter(
                dffs.x,
                dffs.y,
                s=3,
                label=reg,
                facecolor=regimecolors[reg],
                edgecolors="k",
                linewidths=0.1,
            )
        plt.legend(loc=0, title="Erosion regime")

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        plt.axis("equal")
        plt.ylim([df["y"].min(), df["y"].max()])
        plt.show(block=False)

        if "plot_dir" in kwargs:
            plot_dir = kwargs.get("plot_dir")
            figure.savefig(
                os.path.join(plot_dir, "map_erosionregimes"),
                dpi=500,
                bbox_inches="tight",
            )

    def alongshore_sallenger_regimes(
        self,
        zsmean,
        zsmax,
        MHW: float = 0,
        plot_transects: bool = False,
        plot_map: bool = False,
    ):
        """Compute Sallenger and erosion regimes for every alongshore transect.

        Parameters
        ----------
        zsmean : numpy.ndarray
            Mean water surface from XBeach 2-D map,
            shape ``(time, ny, nx)``.
        zsmax : numpy.ndarray
            Maximum water surface from XBeach 2-D map,
            shape ``(time, ny, nx)``.
        MHW : float, optional
            Mean high water level.  Default is ``0``.
        plot_transects : bool, optional
            Plot a subset of individual profiles.  Default is ``False``.
        plot_map : bool, optional
            Generate Sallenger and erosion regime maps.  Default is ``False``.

        Returns
        -------
        x_crest : list of float
            X coordinates of the dune crest for each transect.
        y_crest : list of float
            Y coordinates of the dune crest for each transect.
        regimenos : list of int
            Sallenger regime for each transect.
        erosionregimenos : list of int
            Erosion regime for each transect.
        """
        regimenos = []
        x_crest = []
        y_crest = []
        regimename = []
        erosionregimenos = []
        erosionregimename = []

        for idy in range(0, self.x.shape[0]):
            try:
                distance = euclidean_distance(
                    self.x[idy, -1], self.y[idy, -1], self.x[idy, :], self.y[idy, :]
                )

                # we have to flip the profiles, since beachpy defines profiles land --> ocean
                distance_fl = np.flip(distance)
                zb0_fl = np.flip(self.z0[idy, :])
                zbend_fl = np.flip(self.zend[idy, :])
                zsmax_fl = np.flip(zsmax[:, idy, :])
                zsmean_fl = np.flip(zsmean[:, idy, :])

                # Create profile object using beachpy
                profile = Profile(
                    x=distance_fl,
                    z=zb0_fl,
                    zbend=zbend_fl,
                    ref="MHW",
                    wls=[0.0, MHW],
                    dx=2,
                    id=idy,
                )
                # Get dune crests (foredune and maximum, both pre- and post-storm)
                profile.dune_crest(zmin=MHW, fr_x_min=150, fr_x_max=300)

                # Get sallenger regimes
                try:
                    regimeno, regimena, ind_crest = profile.sallenger_regimes(
                        zsmean_fl, zsmax_fl
                    )
                except Exception:
                    regimeno, regimena = 5, "unknown"
                regimenos.append(regimeno)
                regimename.append(regimena)

                # get erosion regimes
                try:
                    erosionregimeno, erosionregimena, maxeroid = (
                        profile.erosion_regimes()
                    )
                except Exception:
                    erosionregimeno, erosionregimena = 5, "unknown"
                erosionregimenos.append(erosionregimeno)
                erosionregimename.append(erosionregimena)

                # keep in mind that we flipped the profiles first, so need to flip back to get coordinates!!
                ind_crestxy = self.x.shape[1] - ind_crest - 1
                x_crest.append(self.x[idy, ind_crestxy])
                y_crest.append(self.y[idy, ind_crestxy])

                if plot_transects:
                    # plot only a couple of transects
                    if idy in range(0, self.x.shape[0], 15):
                        profile.plot(
                            xlim=[profile.x_dc_fr - 100, profile.x_dc_fr + 200],
                            ylim=[-5, np.nanmax(zb0_fl) + 1],
                            water_levels=[0.0, 0.5],
                            plot_dir=self.plot_dir,
                            plot_show=True,
                        )
            except Exception as e:
                print(str(e))

        self.x_crest = x_crest
        self.y_crest = y_crest
        self.regimenos = regimenos
        self.regimename = regimename
        self.erosionregimenos = erosionregimenos
        self.erosionregimename = erosionregimename

        if plot_map:
            Map.plot_sallengerregimes(self, plot_dir=self.plot_dir)
            Map.plot_erosionregimes(self, plot_dir=self.plot_dir)

        return x_crest, y_crest, regimenos, erosionregimenos


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def euclidean_distance(x1: float, y1: float, x2, y2):
    """Compute Euclidean distance from point (x1, y1) to array of points (x2, y2).

    Parameters
    ----------
    x1 : float
        Reference X coordinate.
    y1 : float
        Reference Y coordinate.
    x2 : array-like
        Target X coordinates.
    y2 : array-like
        Target Y coordinates.

    Returns
    -------
    numpy.ndarray
        Distances from (x1, y1) to each (x2, y2) pair.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def interp_nan(x_or, z_or, dx: float):
    """Interpolate onto a regular grid, ignoring NaN values.

    Parameters
    ----------
    x_or : numpy.ndarray
        Original cross-shore locations.
    z_or : numpy.ndarray
        Original elevations (may contain NaN).
    dx : float
        New grid spacing for interpolation.

    Returns
    -------
    x : numpy.ndarray
        New cross-shore locations.
    z : numpy.ndarray
        Interpolated elevations.
    """
    nonnan = ~np.isnan(z_or)  # find non-nan indices
    x_or, z_or = x_or[nonnan], z_or[nonnan]
    x = np.arange(
        x_or[0], x_or[-1] + dx, dx
    )  # make new grid with first and last non-nan points
    z = np.interp(x, x_or, z_or)  # linearly interpolate elevation values

    return x, z


def mov_average(x, z, win: float):
    """Smooth elevations using a moving-average filter.

    Parameters
    ----------
    x : numpy.ndarray
        Cross-shore locations.
    z : numpy.ndarray
        Elevations.
    win : float
        Window size in metres.

    Returns
    -------
    z_smoothed : numpy.ndarray
        Smoothed elevations.
    """
    n = int(
        np.round(win / (x[1] - x[0]))
    )  # window is found by taking into account x-resolution and window size
    B = 1 / n * np.ones(n)
    z_smoothed = signal.filtfilt(B, 1, z)
    return z_smoothed


def hanning_filter(x, z, window: float):
    """Smooth elevations using a Hanning filter.

    Parameters
    ----------
    x : numpy.ndarray
        Cross-shore locations.
    z : numpy.ndarray
        Elevations.
    window : float
        Window size in metres.

    Returns
    -------
    z_smoothed : numpy.ndarray
        Smoothed elevations.
    """
    window_han = int(
        np.round(window / (x[1] - x[0]))
    )  # window is found by taking into account x-resolution and window size
    with warnings.catch_warnings():  # avoid RuntimeWarning when window is smaller than grid spacing
        warnings.simplefilter("ignore", category=RuntimeWarning)
        han = np.divide(signal.hanning(window_han), np.sum(np.hanning(window_han)))

    z_smoothed = signal.filtfilt(han, 1, z, padtype="even")
    return z_smoothed


def zero_crossing(x, z, indices: bool = False):
    """Find zero-crossings and their indices.

    Adapted from ``PyAstronomy.pyaC.zerocross1d``.

    Parameters
    ----------
    x : numpy.ndarray
        Cross-shore locations.
    z : numpy.ndarray
        Elevations at *x* points.
    indices : bool, optional
        When ``True`` also return the indices of the points just before each
        zero-crossing.  Default is ``False``.

    Returns
    -------
    zz : numpy.ndarray
        Interpolated x values at the zero-crossings.
    zzindi : numpy.ndarray
        Indices of *x* just before each zero-crossing (only returned when
        *indices* is ``True``).
    """
    # Indices of points *before* zero-crossing
    indi = np.where(z[1:] * z[0:-1] < 0.0)[0]
    # Find the zero crossing by linear interpolation
    dx = x[indi + 1] - x[indi]
    dz = z[indi + 1] - z[indi]
    zc = -z[indi] * (dx / dz) + x[indi]
    # What about the points, which are actually zero
    zi = np.where(z == 0.0)[0]
    # Do nothing about the first and last point should they be zero
    zi = zi[np.where((zi > 0) & (zi < x.size - 1))]
    if len(zi) > 1:
        drop = [zi[j + 1] == zi[j] + 1 for j in range(len(zi) - 1)]
        drop = np.concatenate((drop, [False]))
        zi = zi[~drop]
    # Concatenate indices
    zzindi = np.concatenate((indi, zi))
    # Concatenate zc and locations corresponding to zi
    zz = np.concatenate((zc, x[zi]))
    # Sort by x-value
    sind = np.argsort(zz)
    zz, zzindi = zz[sind], zzindi[sind]

    if not indices:
        return zz
    else:
        return zz, zzindi
