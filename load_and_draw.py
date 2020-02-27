from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def load_results(file_name: str = 'saved_ok.bin'):
    sharpe, risk, returns, divers, weights = pickle.load(open(file_name, 'rb'))
    return sharpe, risk, returns, divers, weights
    # USE: sharp, risk, returns, divers, weights = load_results('saved_std.bin')


if __name__ == '__main__':

    files = ['saved_std.bin', 'saved_ok.bin']
    folder = '~/qmes' # add your local directory if needed
    os.chdir(folder)

    for f in files:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sharp, risk, returns, divers, weights = load_results(f)
        img = ax.scatter(sharp, risk, returns, c=divers, cmap=plt.hot())
        fig.colorbar(img)
        plt.show()
        # plt.savefig(f[:-4] + '.png')