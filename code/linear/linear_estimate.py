"""
    A script for exploring linear models
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal     # normal and uniform are random sample generators
from numpy.random import uniform    

# linear function maker
def linear_function_maker(slope_m, bias_b):
    """Return a function, y = f(x), y = m*x + b
    this function binds b and m for a line function
    """
    return lambda x: bias_b + slope_m*x


def plot_normal_distribution(mean=0, sd=1.0, n=1000):
    """Plot a histogram of samples from a normal distribution"""
    sample = normal(loc=mean,scale=sd,size=n)
    plt.hist(sample,density=True)
    plt.show()

    
def main():
    # create 15 evenly spaced values between 0 and 10,000
    # np.linspace
    granularity = 15
    start_x = 0
    end_x = 10000
    X = np.linspace(start_x, end_x, granularity)

    # create a line function, y = (1/50)x + 0
    line_function = linear_function_maker(1/50, 0);
    
    # map the line_function to the evenly spaced x values
    Y = list(map(line_function, X))
    # plot the simple "ideal" model. Perfectly linear with no noise.
    # plt.plot(X,Y,color="green")

    # take 10 samples and add noise
    sample_size_n = 100

    # uniformly sample the input domain (0 to 10,000)
    experiment_X = np.random.uniform(0, 10000, sample_size_n)
    # generate some noise
    error_mean = 0
    error_standard_deviation = 2.5
    # sd = 2.5 means 1 standard deviation is 2.5 pounds.
    # This means that 95% of the variation would be between 2*2.5 on either side of 0
    noise = np.random.normal(error_mean, error_standard_deviation, sample_size_n)
    # calcuate the output of the samples output = [(line model) + noise]
    experiment_Y = list(map(line_function, experiment_X)) + noise

    # plot the noisy observations
    plt.scatter(experiment_X, experiment_Y)
    plt.xlabel("Pounds")
    plt.ylabel("Cubit Feet")

    # perfomr the polynomial fit (degree 1, linear model)
    coeffs = np.polyfit(experiment_X, experiment_Y, 1)            # fit polynomial
    slope,intercept = coeffs
    print("slope=", slope, ", intercept=", intercept)
    predicted = np.polyval(coeffs, X)    # get polynomial value at each x
    plt.plot(X, predicted, color="red")
    plt.show()


main()
