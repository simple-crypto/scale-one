# SCALE-I

This repository contains the jupyter notebook for the SCALE-I training.

## Dependencies

The following dependencies are required in order to run the notebook

+ [Python3](https://www.python.org/downloads/) for installation (version >= 3.10 tested)
+ Python3 venv package (must be installed independently as `python3-venv` on Debian/Ubuntu, installed by default elsewhere)

Besides, the dataset used in this tutorial can be found [here](https://enigma.elen.ucl.ac.be/dsm/sharing/VMx9J3Xkc)

## Getting started

In order to open the notebook, please follow these instructions

1. Clone this repo, containing the framework.
1. [Download](https://enigma.elen.ucl.ac.be/dsm/sharing/VMx9J3Xkc) the dataset and unzip it into the framework directory.
1. Setup the virtual environment and install the dependencies

    + On Linux

        ```shell
        python3 -m venv scale-venv
        ./scale-venv/bin/pip install -r requirements.txt
        ```

    + On Windows 

        ```shell
        py -m venv scale-venv
        .\scale-venv\Scripts\pip install -r requirements.txt
        ```

1. If the previous steps successfully passed, open jupyter lab using the following command

    + On Linux

        ```shell
        ./scale-venv/bin/jupyter-lab
        ```

    + On Windows 

        ```shell
        .\scale-venv\Scripts\jupyter-lab
        ```

Within Jupyter lab (it opens in a web browser), you can now use the file browser (on the left of your screen) to open `scale_one-getting_started.ipynb`, which will introduce you to the basics of Jupyter Lab and checks that everything is properly setup.

The practical sessions require some basic knowledge of the Python programming
language and the NumPy library. If you never used those, here are two very
short tutorials that cover the basics:

- [An Informal Introduction to Python](https://docs.python.org/3/tutorial/introduction.html)
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

