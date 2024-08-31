import numpy as np
import matplotlib.pyplot as plt

FILENAME = "station.csv"


def get_data() -> tuple[np.ndarray, np.ndarray]:
    with open(FILENAME, mode="r") as f:
        lines = f.readlines()

    # remove header line
    lines = lines[1:]

    # first number in each lines is the year, followed by the temp in
    # JAN, then FEB, etc.
    temps = []
    for line in lines:
        if not line:
            continue
        linedata = line.split(",")
        # ignore first entry and 999.90 entries (likely missing data)
        linedata = [
            float(val)
            for i, val in enumerate(linedata)
            if i > 0 and i < 13 and val and val != "999.90"
        ]
        temps.extend(linedata)

    series = np.asarray(temps)
    time = np.arange(len(temps), dtype="float32")

    return time, series


def get_normalized_data() -> tuple[np.ndarray, np.ndarray]:
    time, series = get_data()
    mean = series.mean()
    print("mean", mean)
    series -= mean
    std = series.std(axis=0)
    print("standard dev:", std)
    series /= std
    return (time, series)


if __name__ == "__main__":
    plt.figure("Un-normalized Data")
    plt.plot(get_data()[1])

    plt.figure("Normalized Data")
    plt.plot(get_normalized_data()[1])

    plt.show()
