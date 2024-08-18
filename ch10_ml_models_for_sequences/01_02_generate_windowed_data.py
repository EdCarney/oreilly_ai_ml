import matplotlib.pyplot as plt
import plotext as pltext
import numpy as np


# the idea here is to generate data that we can use to train an ML model for
# make predictions on data sequences; i.e. we would have a series and a label
#
# to do this, we will take our data and break it into windows, where all but
# the final value is the series data, and the final value is the label
#
# we can do this using existing tensorflow tooling to simplify the windowing
# process and data loading

def plot_series_mpl(time, series, mark="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], mark)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_series_txt(time, series, mark="fhd", start=0, end=None):
    pltext.canvas_color("white")
    pltext.plot(time[start:end], series[start:end], color='black', marker=mark)
    pltext.ylim(lower=0)
    pltext.xlabel("Time")
    pltext.ylabel("Value")
    pltext.grid(True)
    pltext.clear_terminal()
    pltext.show()
    input("Press Enter to clear...")
    pltext.clear_terminal()


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype='float32')
baseline = 10
series = trend(time, 0.05)
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# create series
series = baseline + trend(time, slope)\
                  + seasonality(time, period=365, amplitude=amplitude)

series += noise(time, noise_level, seed=42)

plot_series_txt(time, series, mark='braille')
