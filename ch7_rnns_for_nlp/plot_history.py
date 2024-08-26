#! /opt/anaconda3/envs/tensor-flow/bin/python

from typing import Dict, List
import plotext as plt
import json
import sys


def read_hist_from_file(filename: str) -> Dict[str, List[float]]:
    with open(filename, "r") as f:
        hist = json.load(f)
    return hist


def plot_data(title, train_data, validation_data):
    mark = "fhd"
    plt.clear_figure()
    plt.canvas_color("white")
    plt.title(title)
    plt.plot(train_data, label="training", marker=mark, color="black")
    plt.plot(validation_data, label="validation", marker=mark, color="green")
    plt.ylim(lower=0)
    plt.clear_terminal()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: must supply exactly one history file")
        exit(1)

    hist = read_hist_from_file(sys.argv[1])

    plot_data("Accuracy Comparison", hist["acc"], hist["val_acc"])
    input("Showing acc comparison, press any button to show loss comparison")

    plot_data("Loss Comparison", hist["loss"], hist["val_loss"])
    input("Press any button to clear...")

    plt.clear_terminal()
