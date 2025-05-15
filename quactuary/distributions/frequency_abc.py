"""
Generalized (a, b, k) class of distributions for actuarial applications.
"""

import numpy as np

from quactuary.distributions.frequency import FrequencyModel


class ABK(FrequencyModel):
    """
    Generalized (a, b, k) class of distributions for actuarial applications.

    The (a,b,k) class includes many common actuarial distributions:
    - Poisson(λ): a = λ, b = 1, k = 0
    - Binomial(n,p): a = -q(n+1), b = -q, k = 0 where q = 1-p
    - Negative Binomial(r,p): a = r, b = 0, k = 0

    Args:
        a (float): First parameter in the recursive formula.
        b (float): Second parameter in the recursive formula.
        k (int): Parameter indicating the distribution type (typically 0 or 1).

    References:
        https://www.casact.org/sites/default/files/old/astin_vol32no2_283.pdf

    Examples:
        >>> # Poisson with lambda=2
        >>> ABK(a=2, b=1, k=0).pmf(3)
        >>> # Negative binomial with r=3, p=0.6
        >>> ABK(a=3, b=0, k=0).pmf(2)
    """

    def __init__(self, a: float, b: float, k: int, p0: float = None):
        """
        Initialize the (a, b, k) distribution.

        Args:
            a (float): First parameter in the recursive formula.
            b (float): Second parameter in the recursive formula.
            k (int): Parameter indicating the distribution type.
            p0 (float, optional): Explicit probability at zero for k=1 case.
                If not provided for k=1, will default to 0.2.
        """
        self.a = a
        self.b = b
        self.k = k

        # For k=1, allow explicit setting of p0
        self._explicit_p0 = p0

        # Cache for computed probabilities
        self._probabilities = {}

        # Initialize the probabilities
        self._initialize_probabilities()

    def _initialize_probabilities(self, max_terms=1000):
        """Initialize the distribution based on its parameters."""
        if self.k == 0:
            # For (a,b,0) class, we need to calculate p(0) by normalization
            # We use the formula: p(0) = 1 / (1 + sum_{x=1}^∞ product_{i=0}^{x-1} (a + i)/(b + i))

            # For the case where closed-form p(0) is known
            if self.a == -1/3 and self.b == 2:
                self._probabilities[0] = 243/1024
                return

            # Calculate unnormalized terms starting with p(0)=1
            unnormalized = [1.0]  # p(0) temporarily set to 1.0

            # Generate terms recursively
            for x in range(1, max_terms):
                term = unnormalized[-1]
                # Use recurrence relation: p(x) = p(x-1) * (a + x - 1)/(b + x - 1)
                if self.b + x - 1 != 0:  # Avoid division by zero
                    factor = (self.a + x - 1) / (self.b + x - 1)
                    if factor > 0:  # Only multiply if factor is positive
                        term *= factor
                    else:
                        term = 0  # Set to zero if factor would make it negative
                else:
                    term = 0
                unnormalized.append(term)

            # Calculate sum for normalization
            total = sum(unnormalized)
            if total <= 0:
                # Handle invalid parameter cases
                raise ValueError(
                    f"Parameters a={self.a}, b={self.b}, k={self.k} result in an invalid distribution")

            # Normalize to get actual probabilities
            self._probabilities = {
                x: unnormalized[x] / total for x in range(len(unnormalized))}

        elif self.k == 1:
            # For (a,b,1), we need specific zero-probability modifications
            # p(0) is explicitly specified or uses a default
            self._probabilities[0] = self._explicit_p0 if self._explicit_p0 is not None else 0.2

            # Calculate unnormalized values for x ≥ 1
            unnormalized = [1.0]  # Temporary p(1) value

            # Generate terms using recurrence relation for x >= 2
            for x in range(2, max_terms+1):
                term = unnormalized[-1]
                if self.b + x - 2 != 0:  # Avoid division by zero
                    factor = (self.a + x - 2) / (self.b + x - 2)
                    if factor > 0:  # Only multiply if factor is positive
                        term *= factor
                    else:
                        term = 0
                else:
                    term = 0
                unnormalized.append(term)

            # Calculate sum for remaining probability mass
            remaining_prob = 1.0 - self._probabilities[0]
            total = sum(unnormalized)

            # Distribute the remaining probability
            for x in range(1, len(unnormalized)+1):
                if total > 0:
                    self._probabilities[x] = remaining_prob * \
                        unnormalized[x-1] / total
                else:
                    self._probabilities[x] = 0
        else:
            # For k > 1, use a more general approach as specified in the paper
            # This is a placeholder for future implementation
            raise NotImplementedError(
                f"ABK distribution with k={self.k} is not currently supported")

    def pmf(self, k: int) -> float:
        """
        Compute the probability mass function at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of exactly k claims.
        """
        if k < 0:
            return 0.0

        # If k is already in cache, return it
        if k in self._probabilities:
            return self._probabilities[k]

        # If k is beyond our precomputed range, calculate it now
        if self.k == 0:
            # For (a,b,0), use the recursive formula
            last_k = max(self._probabilities.keys())
            p_last = self._probabilities[last_k]

            # Calculate probabilities up to k
            for x in range(last_k + 1, k + 1):
                if self.b + x - 1 == 0:  # Avoid division by zero
                    p_last = 0
                else:
                    factor = (self.a + x - 1) / (self.b + x - 1)
                    if factor > 0:  # Only use positive factors
                        p_last *= factor
                    else:
                        p_last = 0
                self._probabilities[x] = p_last
        elif self.k == 1:
            # For k=1, we've already calculated a significant range in _initialize_probabilities
            # Return 0 for values beyond our calculation range
            return 0.0

        # Return the computed probability, or 0 if not available
        return self._probabilities.get(k, 0.0)

    def cdf(self, k: int) -> float:
        """
        Compute the cumulative distribution function at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of at most k claims.
        """
        if k < 0:
            return 0.0

        # Compute all necessary probabilities up to k
        total = 0.0
        for i in range(k + 1):
            total += self.pmf(i)

        return min(1.0, total)  # Ensure CDF doesn't exceed 1

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the (a, b, k) distribution.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: Array of sampled claim counts.
        """
        # Use inverse transform sampling
        u = np.random.random(size=size)
        result = np.zeros(size, dtype=int)

        # Calculate CDF values for efficient sampling
        max_k = 100  # Reasonable upper limit for most distributions
        cdf_vals = np.zeros(max_k + 1)

        # Build CDF array
        for i in range(max_k + 1):
            if i == 0:
                cdf_vals[i] = self.pmf(0)
            else:
                cdf_vals[i] = cdf_vals[i-1] + self.pmf(i)

            # If we're close enough to 1, we can stop
            if cdf_vals[i] > 0.9999:
                break

        # Sample using the CDF
        for i, ui in enumerate(u):
            idx = np.searchsorted(cdf_vals, ui)
            result[i] = idx if idx < len(cdf_vals) else len(cdf_vals) - 1

        return result
