#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():

    # Loading models:
    with open("history_docker.txt", "rb") as f:
        kubeflow_history = pickle.load(f)

    # print(kubeflow_history.history["accuracy"])

    # plt.plot()

    # plt.show()


if __name__ == "__main__":
    main()
