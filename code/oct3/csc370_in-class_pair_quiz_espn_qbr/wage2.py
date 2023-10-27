import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_stata('WAGE2.DTA')
def linear_function_maker(slope_m, bias_b):
    """Return a function, y = f(x), y = m*x + b
    this function binds b and m for a line function
    lambda (anonymous) function takes in an x and returns bias_b + slope_m*x
    """
    return lambda x: bias_b + slope_m*x


features = ["hours", "educ", "exper", "tenure"]
filtered_col = df.loc[df["wage"] > 700]

# modeling
y_var = df["wage"]
x_var = df["hours"]
coeffs = np.polyfit(x_var, y_var, 1) 
# print(coeffs)

# create 15 evenly spaced values between 0 and 10,000
# basically 14 intervals: 10000/14
granularity = 15
start_x = 0
end_x = 10000
linspace_x = np.linspace(start_x, end_x, granularity)
linear_function = linear_function_maker(0.5, 0) #return lambda x: 0 + 0.5 * x => linear_function takes x as an argument
linspance_y = list(map(linear_function, linspace_x))
# plt.plot(linspace_x, linspance_y,color="green")
# plt.show()


predict_x = np.linspace(min(x_var),max(x_var), 100) 
predicted_y = np.polyval(coeffs, predict_x)  
print(type(predicted_y))
print(type(y_var))
mse = sum((predicted_y - y_var)**2)/len(y_var)
print(mse)
# rmse = np.sqrt(mse)
# print(rmse)
plt.scatter(x_var, y_var)        
plt.plot(predict_x, predicted_y, color="red")
# plt.show()            



# Simple Stats
sample_size_n = 10000
# Create normal-distributed data points
mean = 0
sd = 2.5
normal_x = np.random.normal(mean, sd, sample_size_n)
# print(normal_x.mean())

# Create uniform-distributed data points
lower_bound = 0
higher_bound = 10
uniform_x = np.random.uniform(lower_bound, higher_bound, sample_size_n)
# print(f"mean of uniform-dist should be (high - low) / 2: {uniform_x.mean()}")