import numpy as np

def gaussian_pdf(xs, mean, std):
    coef = 1/(std*np.sqrt((2*np.pi)))
    return coef * np.exp(-(xs-mean)**2 / (2*(std**2)))
