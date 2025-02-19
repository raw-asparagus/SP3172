import numpy as np
from numpy.typing import NDArray


class Knapsack:
    """A class representing the Knapsack Problem.

    This class encapsulates the data and operations for a knapsack problem instance,
    where items with specific values and weights need to be selected to maximize value
    while staying within a weight capacity constraint.

    Attributes:
        _values (NDArray[np.int64]): Array of item values/profits
        _weights (NDArray[np.int64]): Array of item weights
        _capacity (int): Maximum weight capacity of the knapsack
        _num_items (int): Number of available items

    Raises:
        ValueError: If input arrays are empty or have different lengths,
                  or if capacity is negative
        TypeError: If inputs are not of correct types
    """

    def __init__(
            self,
            values: NDArray[np.int64],
            weights: NDArray[np.int64],
            capacity: int
    ) -> None:
        """Initialize the Knapsack problem instance.

        Args:
            values: Array of item values/profits
            weights: Array of item weights
            capacity: Maximum weight capacity

        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs are of wrong type
        """
        # Input validation
        if not isinstance(values, np.ndarray) or not isinstance(weights, np.ndarray):
            raise TypeError("Values and weights must be numpy arrays")
        if not isinstance(capacity, int):
            raise TypeError("Capacity must be an integer")
        if len(values) == 0 or len(weights) == 0:
            raise ValueError("Values and weights arrays cannot be empty")
        if len(values) != len(weights):
            raise ValueError("Values and weights arrays must have the same length")
        if capacity < 0:
            raise ValueError("Capacity cannot be negative")
        if not np.all(weights >= 0):
            raise ValueError("Weights cannot be negative")
        if not np.all(values >= 0):
            raise ValueError("Values cannot be negative")

        self._values: NDArray[np.int64] = values
        self._weights: NDArray[np.int64] = weights
        self._capacity: int = capacity
        self._num_items: int = len(values)

    @property
    def capacity(self) -> int:
        """Get the knapsack capacity.

        Returns:
            int: Maximum weight capacity
        """
        return self._capacity

    @property
    def num_items(self) -> int:
        """Get the number of items.

        Returns:
            int: Number of available items
        """
        return self._num_items

    def get_profit(self, idx: int) -> float:
        """Get the profit/value of a specific item.

        Args:
            idx: Index of the item

        Returns:
            float: Value of the item at the given index

        Raises:
            IndexError: If index is out of bounds
        """
        if not 0 <= idx < self._num_items:
            raise IndexError(f"Index {idx} is out of bounds for {self._num_items} items")
        return float(self._values[idx])

    @property
    def profits(self) -> NDArray[np.int64]:
        """Get all item profits/values.

        Returns:
            NDArray[np.int64]: Array of all item values
        """
        return self._values.copy()

    def get_weight(self, idx: int) -> float:
        """Get the weight of a specific item.

        Args:
            idx: Index of the item

        Returns:
            float: Weight of the item at the given index

        Raises:
            IndexError: If index is out of bounds
        """
        if not 0 <= idx < self._num_items:
            raise IndexError(f"Index {idx} is out of bounds for {self._num_items} items")
        return float(self._weights[idx])

    @property
    def weights(self) -> NDArray[np.int64]:
        """Get all item weights.

        Returns:
            NDArray[np.int64]: Array of all item weights
        """
        return self._weights.copy()

    def __str__(self) -> str:
        """Return a string representation of the Knapsack problem.

        Returns:
            str: Formatted string showing the problem details
        """
        # Title and capacity
        lines = [f"Knapsack Problem (Capacity: {self._capacity})"]

        # Format profits row
        profit_values = [f"{value:3.1f}" for value in self._values]
        lines.append("Profits:  " + "  ".join(profit_values))

        # Format weights row
        weight_values = [f"{weight:3.1f}" for weight in self._weights]
        lines.append("Weights:  " + "  ".join(weight_values))

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Knapsack instance.

        Returns:
            str: Detailed string representation
        """
        return (f"Knapsack(values={self._values}, "
                f"weights={self._weights}, "
                f"capacity={self._capacity})")
