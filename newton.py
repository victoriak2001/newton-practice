def optimize(start, fun, stop_crit=0.0001):
    """
    Run Newton's method to find the optimizer and optimum of a given function.

    Args:
        start (float): The starting value of Newton's method.
        fun (function): The function being optimized.
        stop_crit (float, optional): The stopping criterion for the difference
        between consecutive Newton iterations.

    Returns:
        tuple: The optimizer followed by the optimum of fun as estimated by
        Newton's method.
    """
    # initize the difference in excess of the stopping criterion so the while
    # loop can start
    step_diff = stop_crit + 1
    x_t = start

    while step_diff > stop_crit:
        x_t_plus_one = x_t - deriv(fun)(x_t) / deriv(deriv(fun))(x_t)
        step_diff = abs(x_t_plus_one - x_t)
        x_t = x_t_plus_one

    return (x_t, fun(x_t))


def deriv(fun, epsilon=0.0001):
    """
    Use a difference quotient to estimate the derivative of a given function.

    Args:
        fun (function): The given function.
        epsilon (float, optional): The finite difference used in the difference
        quotient approximation.

    Returns:
        function: An estimate for the first derivative of fun.
    """

    def first_deriv(x):
        return (fun(x + epsilon) - fun(x)) / epsilon

    return first_deriv
