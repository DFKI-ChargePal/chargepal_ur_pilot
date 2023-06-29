from __future__ import annotations
# global
import time

import numpy as np
from threading import Thread
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.backend_bases import Event
from matplotlib.backends.backend_agg import FigureCanvasAgg

# typing
from typing import Iterable
from numpy import typing as npt


class BlitManager:
    def __init__(self, canvas: FigureCanvasAgg, animated_artists: Iterable[Artist] | None = None) -> None:
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists: list[Artist] = []

        if animated_artists:
            for a in animated_artists:
                self._add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event: Event | None) -> None:
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def _add_artist(self, art: Artist) -> None:
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self) -> None:
        """Draw all the animated artists."""
        fig_ = self.canvas.figure
        for a in self._artists:
            fig_.draw_artist(a)

    def _update_screen(self) -> None:
        """Update the screen with animated artists."""
        cv = self.canvas
        fig_ = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self._on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig_.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class SignalMonitor(BlitManager):

    MONITOR_HZ_ = 30

    def __init__(self, y_labels: list[str], hz: int, time_history: float = 1.0) -> None:
        self.display_rate = 1.0/self.MONITOR_HZ_
        self.sig_dim = len(y_labels)
        self._fig, self._axs = plt.subplots(nrows=self.sig_dim)
        self._time_span = np.linspace(-abs(time_history), 0.0, int(time_history * hz))
        self.signals = np.zeros([self.sig_dim, self._time_span.shape[-1]])
        self.y_min = np.zeros(self.sig_dim)
        self.y_max = np.zeros(self.sig_dim)
        self.signal_lns: list[Line2D] = []
        # Create axes plots
        for i, ax_ in enumerate(self._axs):
            # Initialize plots with zeros
            (ln_, ) = ax_.plot(self._time_span, np.zeros_like(self._time_span), animated=True)
            ax_.set_ylabel(y_labels[i])
            ax_.set_xlim(-abs(time_history), 0.0)
            ax_.grid(True)
            self.signal_lns.append(ln_)
        self._axs[-1].set_xlabel('time [s]')
        # Use BlitMangar to update signal data in realtime.
        BlitManager.__init__(self, self._fig.canvas, self.signal_lns)
        # self.bm = BlitManager(self._fig.canvas, self.signal_lns)
        # # Create thread
        # self.thread = Thread(target=self._update_monitor, args=())
        # self.thread.daemon = True
        # self.running = True
        # Make sure our window is on the screen and drawn
        plt.show(block=False)
        plt.pause(.1)
        # Start updating monitor
        # self.thread.start()

    def _update_monitor(self) -> None:
        y_min, y_max = np.min(self.signals, axis=-1), np.max(self.signals, axis=-1)
        y_min_idx = np.where(y_min < self.y_min)
        y_max_idx = np.where(y_max > self.y_max)
        self.y_min[y_min_idx] = y_min[y_min_idx]
        self.y_max[y_max_idx] = y_max[y_max_idx]
        for idx, ln_ in enumerate(self.signal_lns):
            ln_.set_ydata(self.signals[idx, :])
            y_min_, y_max_ = self.y_min[idx], self.y_max[idx]
            if np.abs(y_min_) + np.abs(y_max_) <= 0.1:
                y_min_ -= 0.1
                y_max_ += 0.1
            ln_.axes.set_ylim(1.1 * y_min_, 1.1 * y_max_)
        self._update_screen()

    def add(self, signal: npt.NDArray[np.float64]) -> None:
        sig_dim, sig_len = signal.shape
        if sig_dim != self.sig_dim:
            raise ValueError(f'Given signal do not match with the number of plots. {sig_dim} != {self.sig_dim}')
        self.signals = np.hstack([self.signals[:, sig_len:], signal])
        self._update_monitor()


    # def destroy(self) -> None:
    #     self.running = False
    #     self.thread.join()
