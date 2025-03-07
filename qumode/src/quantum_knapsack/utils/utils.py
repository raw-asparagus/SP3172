import time
from functools import wraps
from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray

# Constants
DECIMAL_PRECISION = 3
COLUMN_DELIMITER = "   "
MATRIX_BORDERS = {
    'top_left': '⌈',
    'top_right': '⌉',
    'bottom_left': '⌊',
    'bottom_right': '⌋',
    'middle': '|'
}


def timer_decorator(func: Callable) -> Callable:
    """Decorator to measure and print function execution time.

    Args:
        func: The function to be timed.

    Returns:
        Callable: Wrapped function that prints execution time.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print(f"'{func.__name__}' executed in:\t{execution_time:.4f}s")
        return result

    return wrapper


def _format_complex(complex_num: Union[complex, np.complex128, np.ndarray, float, int]) -> str:
    """Format a single complex number as a string for LaTeX.

    Args:
        complex_num: Complex number to format (can be real or complex).

    Returns:
        str: LaTeX formatted string representation of the number.
    """
    # Handle NumPy array case
    if isinstance(complex_num, np.ndarray):
        if complex_num.size != 1:
            raise ValueError("Input array must contain exactly one element")
        complex_num = complex_num.item()

    # Convert to complex number if it's a real number
    if isinstance(complex_num, (float, int)):
        return f"{float(complex_num):.3f}"

    # Extract real and imaginary parts
    real = float(complex_num.real)
    imag = float(complex_num.imag)

    # Format for LaTeX
    if imag == 0:
        return f"{real:.3f}"
    elif real == 0:
        return f"{imag:.3f}i"
    else:
        sign = " - " if imag < 0 else " + "
        return f"{real:.3f}{sign}{abs(imag):.3f}i"


def pretty_format(matrix: NDArray[np.complex128], matrix_type: str = 'bmatrix') -> str:
    """Convert a numpy array to LaTeX matrix representation.

    Args:
        matrix: 2D numpy array of numbers (real or complex).
        matrix_type: LaTeX matrix environment type ('bmatrix', 'pmatrix', 'matrix', etc.).

    Returns:
        str: LaTeX code for the matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")

    # Start the LaTeX matrix
    latex_str = f"\\begin{{{matrix_type}}}\n"

    # Format each row
    rows = []
    for row in matrix:
        # Format each element in the row and join with LaTeX column separator
        row_str = " & ".join(_format_complex(elem) for elem in row)
        rows.append(row_str)

    # Join rows with LaTeX row separator
    latex_str += " \\\\\n".join(rows)

    # Close the matrix environment
    latex_str += f"\n\\end{{{matrix_type}}}"

    return r'' + latex_str
