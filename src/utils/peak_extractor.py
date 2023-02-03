import json
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from tqdm import tqdm

from engine.audio import load_audio
from engine.features.feature_extraction import FeaturesFileList
from engine.helpers import read_file, write_file
from engine.metadata import metadata_db
from engine.nn.properties import ModelProperties
from engine.settings import BIRDVOICE_FOLDER
from engine.utils import copy

IMG_HEIGHT = 256
XPPS = 2000
# SAMPLING_RATE = 32000
SAMPLING_RATE = 500000
# LOW_FILTER = 4750
LOW_FILTER = 1000
# HIGH_FILTER = 8500
HIGH_FILTER = 48000
STRETCH = 10

HIGH_FREQ = SAMPLING_RATE / 2
LOW_PX, HIGH_PX = int((1 - LOW_FILTER / HIGH_FREQ) * 256), int((1 - HIGH_FILTER / HIGH_FREQ) * 256)
PEAK_PAD = XPPS // 2

print(LOW_PX, HIGH_PX)


# Good example for no: ind33_2018_06_16_0434_MON_LG_33_1_2
# Good example with messy data: 2017_06_23_ROC_LG_19_01_01


def moving_average(a, n=3):
    """https://stackoverflow.com/a/14314054"""
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@dataclass
class Context:
    cdt_peak: int
    cdt_line_s: Line2D
    cdt_line_p: Line2D
    ok_peaks: List[float]
    stop: bool


def run(metadata, source):
    mdb = metadata_db(metadata)
    json_data = json.loads(read_file(BIRDVOICE_FOLDER / metadata))
    dest_file = BIRDVOICE_FOLDER / metadata
    copy(dest_file, dest_file.with_suffix('.bak.json'))
    print(f'Destination: {dest_file}')
    load_audio().noise_reduce()
    audio = load_audio(mdb, source).create_spectrogram(
        height=IMG_HEIGHT, light_background=False, x_pixels_per_sec=XPPS, sampling_rate=SAMPLING_RATE)
    ff: FeaturesFileList = audio.img_features().run()

    mp = ModelProperties(ff, prepare_args=dict(add_dim=False, normalize_samples=False), fit_args={})
    x, _, _ = mp.get_features_xyw(variable_length=True)

    for i, f in zip(x, tqdm(ff.files)):
        name = str(f.p.stem)

        if 'peaks' in f.metadata:
            print(f'Already have info for {name}')
            continue

        i = (1 - i) * -120  # Convert to dBFS
        i_filtered = i[:, HIGH_PX:LOW_PX]
        amplitude = np.average(i_filtered, axis=1)
        fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
        axes[0].plot(np.array(range(amplitude.shape[0])) / XPPS, amplitude, ':', label='amplitude')
        axes[0].set_xlabel('Time [s]')
        axes[0].set_ylabel('Amplitude in frequency range [dBFS]')
        windows = [10, 20, 30, 50]
        amplitude_ma = [(w, moving_average(amplitude, w)) for w in windows]
        for w, ma in amplitude_ma:
            axes[0].plot(np.array(range(w // 2, ma.shape[0] + w // 2)) / XPPS, ma,
                         label=f'average {1000 * w // XPPS} ms')
        # axes[0].plot(range(c.shape[0]), np.min(c, axis=1), label='min')
        axes[0].legend()

        im = axes[1].imshow(i.transpose(), cmap='seismic', interpolation='bicubic', aspect=.05,
                            extent=(0, i.shape[0] * STRETCH / XPPS, 0, HIGH_FREQ / 1000))
        axes[1].axhline(LOW_FILTER / 1000, color='#cccc00')
        axes[1].axhline(HIGH_FILTER / 1000, color='#cccc00')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Frequency [kHz]')

        w, ma = amplitude_ma[1]
        ma = np.ma.array(ma, mask=False)

        c_bar = fig.colorbar(im, pad=.01)
        c_bar.ax.set_ylabel('Amplitude [dBFS]')

        fig.suptitle(name)
        fig.show()

        c = Context(
            cdt_peak=w // 2 + np.argmax(ma),
            cdt_line_p=axes[0].axvline((w // 2 + np.argmax(ma)) / XPPS, color='#00cc00'),
            cdt_line_s=axes[1].axvline((w // 2 + np.argmax(ma)) / XPPS * STRETCH, color='#00cc00'),
            ok_peaks=[],
            stop=False
        )

        exit_ax = plt.axes((.93, 0.01, 0.065, 0.05))
        exit_btn = Button(exit_ax, 'Exit')
        skip_ax = plt.axes((.93, 0.07, 0.065, 0.05))
        skip_btn = Button(skip_ax, 'Skip')
        save_ax = plt.axes((.93, 0.13, 0.065, 0.05))
        save_btn = Button(save_ax, 'Save')
        save_ax.set_visible(False)
        no_ax = plt.axes((.93, 0.19, 0.065, 0.05))
        no_btn = Button(no_ax, 'No')
        yes_ax = plt.axes((.93, 0.25, 0.065, 0.05))
        yes_btn = Button(yes_ax, 'Yes')
        minus_ax = plt.axes((.93, 0.31, 0.030, 0.05))
        minus_btn = Button(minus_ax, '<')
        plus_ax = plt.axes((.965, 0.31, 0.030, 0.05))
        plus_btn = Button(plus_ax, '>')

        def more_peaks(c: Context):
            if len(c.ok_peaks) == 2:
                return False
            return ma.shape[0] > 3 * XPPS

        def next_peak(c: Context):
            ma.mask[max(0, c.cdt_peak - PEAK_PAD):min(ma.shape[0], c.cdt_peak + PEAK_PAD)] = True
            c.cdt_peak = w // 2 + np.argmax(ma)
            c.cdt_line_p.remove()
            c.cdt_line_s.remove()
            c.cdt_line_p = axes[0].axvline(c.cdt_peak / XPPS, color='#00cc00')
            c.cdt_line_s = axes[1].axvline(c.cdt_peak / XPPS*STRETCH, color='#00cc00')

        def move_peak(c: Context, offset: int):
            c.cdt_peak += offset
            c.cdt_line_p.remove()
            c.cdt_line_s.remove()
            c.cdt_line_p = axes[0].axvline(c.cdt_peak / XPPS, color='#00cc44')
            c.cdt_line_s = axes[1].axvline(c.cdt_peak / XPPS*STRETCH, color='#00cc44')
            plt.draw()

        def yes_act(_):
            tqdm.write(f'Peak {c.cdt_peak} is valid')
            c.ok_peaks.append(c.cdt_peak / XPPS*STRETCH)
            if more_peaks(c):
                axes[0].axvline(c.cdt_peak / XPPS, color='#000000')
                axes[1].axvline(c.cdt_peak / XPPS*STRETCH, color='#000000')
                next_peak(c)
            else:
                yes_ax.set_visible(False)
                no_ax.set_visible(False)
                minus_ax.set_visible(False)
                plus_ax.set_visible(False)
                save_ax.set_visible(True)
            plt.draw()

        def no_act(_):
            axes[0].axvline(c.cdt_peak / XPPS, color='#cc0000')
            axes[1].axvline(c.cdt_peak / XPPS*STRETCH, color='#cc0000')
            tqdm.write(f'Peak {c.cdt_peak} is invalid')
            next_peak(c)
            plt.draw()

        def save_act(_):
            if more_peaks(c):
                return
            tqdm.write(f'Save {name}: {c.ok_peaks}')
            json_data[name]['peaks'] = sorted(c.ok_peaks)
            write_file(dest_file, json.dumps(json_data, indent='\t'))
            plt.close()

        def skip_act(_):
            tqdm.write(f'Skip {name}')
            plt.close()

        def click_act(evt: MouseEvent):
            if evt.inaxes not in axes:
                return
            c.cdt_peak = int(evt.xdata * XPPS*STRETCH)
            c.cdt_line_p.remove()
            c.cdt_line_s.remove()
            c.cdt_line_p = axes[0].axvline(c.cdt_peak / XPPS, color='#00cc00')
            c.cdt_line_s = axes[1].axvline(c.cdt_peak / XPPS*STRETCH, color='#00cc00')
            plt.draw()

        def exit_act(_):
            c.stop = True
            plt.close()

        yes_btn.on_clicked(yes_act)
        no_btn.on_clicked(no_act)
        save_btn.on_clicked(save_act)
        skip_btn.on_clicked(skip_act)
        exit_btn.on_clicked(exit_act)
        minus_btn.on_clicked(lambda _: move_peak(c, -1))
        plus_btn.on_clicked(lambda _: move_peak(c, +1))
        fig.canvas.mpl_connect('button_press_event', click_act)

        fig.canvas.set_window_title('Bird Voice Peak Extractor')

        plt.show()
        print('next round')
        if c.stop:
            break


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Helps extracting the peak signals from the samples\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('source', help='Audio files list')
    parser.add_argument('-m', '--metadata', metavar='DB', default='metadata.json', help='Metadata database')
    args = parser.parse_args()
    run(**vars(args))
