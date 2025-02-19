import time
from functools import wraps
from numbers import Complex
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
    """Format a single complex number as a string.

    Args:
        complex_num: Complex number to format. Can be:
            - Python complex
            - NumPy complex128
            - NumPy ndarray containing a single complex number
            - Real numbers (float/int which will be treated as complex with zero imaginary part)

    Returns:
        str: Formatted string representation of the complex number.

    Raises:
        TypeError: If input is not a valid complex number type.
        ValueError: If input is a ndarray with more than one element.
    """
    # Handle NumPy array case
    if isinstance(complex_num, np.ndarray):
        if complex_num.size != 1:
            raise ValueError("Input array must contain exactly one element")
        complex_num = complex_num.item()

    # Check if input is a valid numeric type that can be treated as complex
    if not isinstance(complex_num, (Complex, np.complex128, float, int)):
        raise TypeError(
            f"Input must be a complex number or real number, got {type(complex_num)}"
        )

    # Convert to complex number if it's a real number
    if isinstance(complex_num, (float, int)):
        complex_num = complex(complex_num)

    # Extract real and imaginary parts
    real = float(complex_num.real)
    imag = float(complex_num.imag)

    # Format the string representation
    sign = " - " if imag < 0 else " + "
    return f"{real:.{DECIMAL_PRECISION}f}{sign}{abs(imag):.{DECIMAL_PRECISION}f}i"


def pretty_format(matrix: NDArray[np.complex128]) -> str:
    """Pretty print a numpy array of complex numbers.

    Args:
        matrix: 2D numpy array of complex numbers.

    Returns:
        str: Formatted string representation of the matrix.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If input is not a 2D array.

    Example:
        >>> arr = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
        >>> print(pretty_format(arr))
        ⌈   1.000 + 2.000i   3.000 + 4.000i   ⌉
        |   5.000 + 6.000i   7.000 + 8.000i   |
        ⌊   5.000 + 6.000i   7.000 + 8.000i   ⌋
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")

    rows, cols = matrix.shape
    formatted_matrix = [['' for _ in range(cols)] for _ in range(rows)]
    col_widths = [0] * cols

    # Format each number and find maximum width for each column
    for i in range(rows):
        for j in range(cols):
            formatted_matrix[i][j] = _format_complex(matrix[i, j])
            col_widths[j] = max(col_widths[j], len(formatted_matrix[i][j]))

    # Build the formatted string
    result = []
    for i in range(rows):
        # Add left border
        if i == 0:
            row = [MATRIX_BORDERS['top_left'] + COLUMN_DELIMITER]
        elif i == rows - 1:
            row = [MATRIX_BORDERS['bottom_left'] + COLUMN_DELIMITER]
        else:
            row = [MATRIX_BORDERS['middle'] + COLUMN_DELIMITER]

        # Add formatted numbers
        row.extend(
            f"{formatted_matrix[i][j]:>{col_widths[j]}}"
            for j in range(cols)
        )

        # Add right border
        if i == 0:
            row.append(f"{COLUMN_DELIMITER}{MATRIX_BORDERS['top_right']}")
        elif i == rows - 1:
            row.append(f"{COLUMN_DELIMITER}{MATRIX_BORDERS['bottom_right']}")
        else:
            row.append(f"{COLUMN_DELIMITER}{MATRIX_BORDERS['middle']}")

        result.append(COLUMN_DELIMITER.join(row))

    return "\n".join(result)
