# take a starting value, a function, and (optional) a stopping criterion for
# Newton's method. 
# return a tuple with Newton's method estimates for the optimizer and optimum
# of the given function.
def optimize(start, fun, stop_crit=0.0001):
    # initize the difference in excess of the stopping criterion so the while
    # loop can start
    step_diff = stop_crit + 1 
    x_t = start

    while(step_diff > stop_crit):
        x_t_plus_one = x_t - deriv(fun)(x_t) / deriv(deriv(fun))(x_t)
        step_diff = abs(x_t_plus_one - x_t)
        x_t = x_t_plus_one
    
    return (x_t, fun(x_t))

# take a function and (optional) a finite difference epsilon.
# return a finite difference approximation of its first derivative.
def deriv(fun, epsilon=0.0001):
    def first_deriv(x):
        return (fun(x + epsilon) - fun(x)) / epsilon
    return first_deriv