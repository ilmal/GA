#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json


def main():

    # Loading models:
    # with open("history_docker.txt", "rb") as f:
    #     history = pickle.load(f)

    history = json.load(open("history_docker.txt", 'r'))

    print(history)

    print(history.params)

    # plt.plot()

    # plt.show()


if __name__ == "__main__":
    main()
