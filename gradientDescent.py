import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def computeCost(b, m, x, y, df):
    totalError = 0
    for i in range(0, len(df)):
        totalError += np.square(y[i] - (m*x[i]) + b)
        #print(totalError)
    return totalError/len(df)

def gradientStep(this_b, this_m, df, alpha, x, y):
    grad_b = 0
    grad_m = 0
    m = len(df)
    for i in range(0, m):
        grad_b += -(2/m) * (y[i] - ((this_m * x[i]) + this_b))
        grad_m += -(2/m) * x[i] * (y[i] - ((this_m * x[i]) + this_b))
    new_b = this_b - (alpha * grad_b)
    new_m = this_m - (alpha * grad_m)
    return [new_b, new_m]

def gradient_descent_runner(df, starting_b, starting_m, alpha, num_iterations, x, y):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = gradientStep(b, m, df, alpha, x, y)
        m = np.float64(m).item()
        b = np.float64(b).item()
        plt.plot(np.asarray(x), m*np.asarray(x)+b)
   
def run():
    d = pd.read_csv('earthquake.csv')
    df = pd.DataFrame(data=d)
    df = df[['country','richter', 'mw']]
    df = df[df['country']=='mediterranean']
    df.dropna(inplace=True)
    x = list(df['richter'].values)
    y = list(df['mw'].values)
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    alpha = 0.001
    num_iterations = 600
    plt.scatter(x, y)
    gradient_descent_runner(df, initial_b, initial_m, alpha, num_iterations, x, y)
    plt.show()

    
    
if __name__ == '__main__':
    run()