# SCALE-I

This repository contains the jupyter notebook for the SCALE-I training.

## Dependencies

The following dependencies are required in order to run the notebook

+ [Python3](https://www.python.org/downloads/) for installation (version >= 3.10 tested)
+ Python3 venv package (must be installed independently on Debian/Ubuntu, installed by default elsewhere)

Besides, the dataset used in this tutorial can be found [here](https://enigma.elen.ucl.ac.be/dsm/sharing/VMx9J3Xkc)

## Getting started

In order to open the notebook, please follow these instructions

1. Clone this repo, containing the framework.
1. [Download](https://enigma.elen.ucl.ac.be/dsm/sharing/VMx9J3Xkc) the dataset and unzip it into the framework directory.
1. Setup the virtual environment and install the dependencies

    + On Linux

        ```shell
        python3 -m venv scale-venv
        source scale-venv/bin/activate
        ```

    + On Windows 

        ```shell
        py -m venv scale-venv
        .\scale-venv\Scripts\activate
        ```

1. Install the dependencies

    ```shell
    pip install -r requirements.txt
    ```

1. If the previous steps successfully passed, open the jupyter notebook using the following command

    ```shell
    # Link the virtual environment to the notebook
    python -m ipykernel install --user --name=scale-venv
    # Start the notebook
    python -m jupyter notebook simple-scale.ipynb
    ```
