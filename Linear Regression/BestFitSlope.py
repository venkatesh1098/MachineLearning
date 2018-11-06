# slope formulae
# m =  (x' *  y'  - (x*y)')
#      ----------------------
#       ((x')^2) - ((x^2)')
# " ' represents mean" 
# y-intercept = y' - mx'
#m=slope


import statistics 
import numpy as np
import matplotlib.pyplot as plt
import random 

from matplotlib import style
style.use('fivethirtyeight')

#commenteed out for including random datasets
# xs = np.array([1,2,3,4,5,6],dtype=np.float64)
# ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def create_datasets(how_many, variance , step=2, correlation=False):#correlation if true value =Positive False =Negative
    val = 1 
    ys = []
    for i in range(how_many):
        y= val +random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]   
           
    return np.array(xs,dtype= np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope(xs,ys) :
    x_compliment= statistics.mean(xs)
    y_compliment = statistics.mean(ys)
    x_into_y_compliment = statistics.mean(xs*ys)
    square_of_compliment = x_compliment*x_compliment
    compliment_of_square = statistics.mean(xs*xs)
#slope = m
    m = ((x_compliment*y_compliment)-x_into_y_compliment)/(square_of_compliment-compliment_of_square)
    # m = (( (statistics.mean(xs) * statistics.mean(ys)) - statistics.mean(xs*ys)) / (statistics.mean(xs)*statistics.mean(xs)) - statistics.mean(xs*xs))
    return m , b

def best_fit_slope_and_intercept(xs,ys) :
    x_compliment= statistics.mean(xs)
    y_compliment = statistics.mean(ys)
    x_into_y_compliment = statistics.mean(xs*ys)
    square_of_compliment = x_compliment*x_compliment
    compliment_of_square = statistics.mean(xs*xs)
#slope = m |||| c = y-intercept
    m = ((x_compliment*y_compliment)-x_into_y_compliment)/(square_of_compliment-compliment_of_square)
    # m = (( (statistics.mean(xs) * statistics.mean(ys)) - statistics.mean(xs*ys)) / (statistics.mean(xs)*statistics.mean(xs)) - statistics.mean(xs*xs))
    c = y_compliment - (m * x_compliment)
    return m , c 

def squared_error(ys_origin,ys_line):
    return sum((ys_line-ys_origin)**2) 

def coeffecient_of_determination(ys_origin,ys_line):
    y_mean_line = [statistics.mean(ys_origin)for y in ys_origin]
    squared_error_regression = squared_error(ys_origin,ys_line) 
    squared_error_y_mean = squared_error(ys_origin,y_mean_line)
    return 1-(squared_error_regression/squared_error_y_mean)


xs,ys = create_datasets(40, 40, 2, correlation = 'pos')

m,c = best_fit_slope_and_intercept(xs,ys)
#y=mx+c
regression_line = [(m*x)+c for x in xs]

predict_x = 8
predict_y  = (m*predict_x)+c

r_squared = coeffecient_of_determination(ys,regression_line) 
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs, regression_line)
plt.show()
# print(m,c)



# PEMDAS 
# P - Parenthesis
# E - Expponential
# M - Multiplication
# D - Division
# A - Addition
# S - Subtraction
# Order of Mathematical Operation