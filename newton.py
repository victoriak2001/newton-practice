import numpy.linalg as npla
import numdifftools as nd
import scipy.linalg as scila

def optimize(start, fun, stop_crit=1e-5):
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
        str: A brief description of why Newton's method failed, if applicable

    Example:
        >>> import newton
        >>> import numpy as np
        >>> newton.optimize(1, np.sin)
        (np.float64(1.5707463267949306), np.float64(0.99999999875))
        # this is very close to sine's local maximum at (pi / 2, 1)
    """
    if not callable(fun):
        raise TypeError(f"Argument fun is not a function. \
                        It is of type {type(fun)}.")

    if type(start) is not int and type(start) is not float:
        raise TypeError(f"Argument start is not an integer or a float. \
        It is of type {type(start)}.")

    # initize the difference in excess of the stopping criterion so the while
    # loop can start
    step_diff = stop_crit + 1
    x_t = start

    while step_diff > stop_crit:
        x_t_plus_one = x_t - deriv(fun)(x_t) / deriv(deriv(fun))(x_t)
        step_diff_new = abs(x_t_plus_one - x_t)
        x_t = x_t_plus_one

        if step_diff_new > 2 * step_diff:
            return "Newton's method failed to converge"
        step_diff = step_diff_new

        if x_t > 1e8:
            raise RuntimeError("Optimization appears to be diverging")

    return (x_t, fun(x_t))


def deriv(fun, epsilon=1e-5):
    """
    Use a difference quotient to estimate the derivative of a given function.

    Args:
        fun (function): The given function.
        epsilon (float, optional): The finite difference used in the difference
        quotient approximation.

    Returns:
        function: An estimate for the first derivative of fun.

    Example:
        >>> import newton
        >>> import numpy as np
        >>> sin_prime = newton.deriv(np.sin) # this should approximate cosine
        >>> sin_prime(0)
        np.float64(0.9999999983333334)
        # this is very close to cos(0) = 1
    """

    def first_deriv(x):
        return (fun(x + epsilon) - fun(x)) / epsilon

    return first_deriv

def optimize_multivar(start, fun, stop_crit=1e-5):
    """
    Run Newton's method to find the optimizer and optimum of a given 
    multivariate function.

    Args:
        start (list): The starting value of Newton's method.
        fun (function): The function being optimized.
        stop_crit (float, optional): The stopping criterion for the difference
        between consecutive Newton iterations.

    Returns:
        tuple: The optimizer followed by the optimum of fun as estimated by
        Newton's method.
        str: A brief description of why Newton's method failed, if applicable

    Example:
    """
    if not callable(fun):
        raise TypeError(f"Argument fun is not a function. \
                        It is of type {type(fun)}.")

    # initize the difference in excess of the stopping criterion so the while
    # loop can start
    step_diff = stop_crit + 1
    x_t = start

    while step_diff > stop_crit:
        h = nd.Hessian(fun)(x_t)
        g = nd.Gradient(fun)(x_t)

        x_t_plus_one = x_t - scila.solve(h, g)
        step_diff_new = npla.norm(x_t_plus_one - x_t)
        x_t = x_t_plus_one

        if step_diff_new > 2 * step_diff:
            return "Newton's method failed to converge"
        step_diff = step_diff_new

        if npla.norm(x_t) > 1e8:
            raise RuntimeError("Optimization appears to be diverging")

    return (x_t, fun(x_t))
