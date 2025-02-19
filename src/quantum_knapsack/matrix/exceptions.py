class MatrixError(Exception):
    """Base exception class for matrix-related errors."""
    pass


class DegenerateException(MatrixError):
    """Exception raised when a matrix is degenerate."""
    pass


class ObservableException(MatrixError):
    """Exception raised when there's an issue with an Observable matrix."""
    pass
