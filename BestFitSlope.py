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

xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def best_fit_slope(xs,ys) :
    x_compliment= statistics.mean(xs)
    y_compliment = statistics.mean(ys)
    x_into_y_compliment = statistics.mean(xs*ys)
    square_of_compliment = x_compliment*x_compliment
    compliment_of_square = statistics.mean(xs*xs)
#slope = m
    m = ((x_compliment*y_compliment)-x_into_y_compliment)/(square_of_compliment-compliment_of_square)
    # m = (( (statistics.mean(xs) * statistics.mean(ys)) - statistics.mean(xs*ys)) / (statistics.mean(xs)*statistics.mean(xs)) - statistics.mean(xs*xs))
    return m

hello = best_fit_slope(xs,ys)
print(hello)



# PEMDAS 
# P - Parenthesis
# E - Expponential
# M - Multiplication
# D - Division
# A - Addition
# S - Subtraction
# Order of Mathematical Operation