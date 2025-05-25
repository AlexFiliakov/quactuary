"""
Insurance entity models for policy terms, inforce buckets, and portfolios.

This module defines the core business objects for representing insurance policy
terms (`PolicyTerms`), aggregation buckets (`Inforce`), and collections of buckets (`Portfolio`).

Examples:
    >>> from quactuary.entities import PolicyTerms, Inforce, Portfolio
    >>> terms = PolicyTerms(per_occ_deductible=100.0, coinsurance=0.8)
    >>> bucket = Inforce(n_policies=50, freq=freq_model, sev=sev_model, terms=terms)
    >>> portfolio = Portfolio([bucket])
"""
import locale
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quactuary.backend import BackendManager, set_backend
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.distributions.qmc_wrapper import wrap_for_qmc
from quactuary.sobol import get_qmc_simulator

locale.setlocale(locale.LC_ALL, '')


@dataclass(frozen=True, slots=True)
class ExposureBase:
    """
    Base class for exposure definitions.

    Attributes:
        name (str): Name of the exposure base.
        description (str): Description of the exposure base.
    """
    name: str
    unit: str
    description: str


# Predefined exposure bases
PAYROLL = ExposureBase(name="Payroll", unit="USD",
                       description="Total payroll amount.")
SALES = ExposureBase(name="Sales", unit="USD",
                     description="Total sales amount.")
SQUARE_FOOTAGE = ExposureBase(
    name="Square Footage", unit="sq ft", description="Total square footage.")
VEHICLES = ExposureBase(name="Vehicles", unit="count",
                        description="Total number of vehicles.")
REPLACEMENT_VALUE = ExposureBase(
    name="Replacement Value", unit="USD", description="Total replacement value.")
NON_INFLATIONARY_UNIT = ExposureBase(
    name="Non-Inflationary Unit", unit="unit", description="Non-inflationary unit of measure.")


class LOB(str, Enum):
    """
    Enumeration of lines of business (LOB) for insurance policies.
    """
    GLPL = "General and Product Liability"
    GL = "General Liability"
    PL = "Product Liability"
    WC = "Workers' Compensation"
    CAuto = "Auto Liability"
    EL = "Employers' Liability"
    D_O = "Directors and Officers Liability"
    E_O = "Errors and Omissions Liability"
    CPP = "Commercial Property Package"
    Cyber = "Cyber Liability"
    EPLI = "Employment Practices Liability"
    Umbrella = "Umbrella Liability"
    PAuto = "Personal Auto"
    PProperty = "Personal Property"
    PPL = "Personal Property Liability"
    PGL = "Personal General Liability"
    PWC = "Personal Watercraft"
    PWCPL = "Personal Watercraft Liability"

    def __str__(self):
        return self.value


@dataclass
class PolicyResult:
    """
    Detailed breakdown of policy term application.
    
    Attributes:
        ground_up_loss: Original loss amount
        retained_amount: Amount retained by insured
        loss_after_retention: Loss after applying retention
        excess_attachment: Amount below XoL attachment (if applicable)
        loss_after_attachment: Loss after XoL attachment
        limited_loss: Loss after applying limits
        amount_over_limit: Amount exceeding limits
        corridor_retained: Amount retained by corridor
        insurer_share: Final amount paid by insurer after coinsurance
        insured_share: Final amount paid by insured (retention + coinsurance)
    """
    ground_up_loss: float
    retained_amount: float
    loss_after_retention: float
    excess_attachment: float
    loss_after_attachment: float
    limited_loss: float
    amount_over_limit: float
    corridor_retained: float
    insurer_share: float
    insured_share: float


@dataclass(frozen=True, slots=True)
class PolicyTerms:
    """
    Terms defining insurance policy layer details.

    Attributes:
        per_occ_deductible (float): Deductible per occurrence.
        coinsurance (float): Insurer share proportion (1.0 = 100%).
        per_occ_limit (Optional[float]): Per-occurrence limit.
        agg_limit (Optional[float]): Aggregate limit.
        attachment (float): Attachment point for excess-of-loss.
        coverage (str): Coverage type (e.g., "OCC", "CLAIMS-MADE").
    """
    effective_date:     date  # Effective date of the policy
    expiration_date:    date  # Expiration date of the policy
    lob:                Optional[LOB] = None  # Line of business (optional)
    # Exposure base (Payroll, Sales, Square Footage, Vehicles, Replacement Value etc.)
    exposure_base:      Optional[ExposureBase] = None
    # Exposure amount (e.g., limit, sum insured)
    exposure_amount:    float = 0.0
    retention_type:     str = "deductible"  # deductible / SIR
    per_occ_retention:  float = 0.0  # retention per occurrence
    agg_retention:      Optional[float] = None  # aggregate retention
    corridor_retention: Optional[float] = None  # corridor retention
    # 0 → 0% insured's share, 1.0 → 100% insured's share
    coinsurance:        Optional[float] = None
    # per-occurrence limit (None if unlimited)
    per_occ_limit:      Optional[float] = None
    agg_limit:          Optional[float] = None  # aggregate limit (Optional)
    attachment:         Optional[float] = None  # for XoL layers
    coverage:           str = "occ"  # occ (occurrence) / cm (claims-made)
    notes:              str = ""  # Additional notes (ad hoc)
    # ...additional fields defined as needed...
    # TODO: reinstatements: Reinstatement terms (optional).
    # TODO: corridors: Corridor terms (optional).
    # TODO: premium: Premium amount (optional).
    # TODO: commission: Commission percentage (optional).
    # TODO: taxes: Taxes applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: exclusions: Exclusions applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: sublimits: Sublimits applicable to the policy (optional).

    def __str__(self) -> str:
        output = f"Effective Date: {self.effective_date}\n" + \
            f"Expiration Date: {self.expiration_date}\n"
        if self.lob:
            output += f"LoB: {self.lob}\n"
        if self.exposure_base:
            output += f"Exposure Base: {self.exposure_base}\n"
        output += f"Exposure Amount: {self.exposure_amount:n}\n"
        output += f"Retention Type: {self.retention_type}\n"
        output += f"Per-Occurrence Retention: {self.per_occ_retention:n}\n"
        if self.agg_retention:
            output += f"Aggregate Retention: {self.agg_retention:n}\n"
        if self.corridor_retention:
            output += f"Corridor Retention: {self.corridor_retention:n}\n"
        if self.coinsurance:
            output += f"Coinsurance: {self.coinsurance:.2%}\n"
        if self.per_occ_limit:
            output += f"Per Occurrence Limit: {self.per_occ_limit:n}\n"
        if self.agg_limit:
            output += f"Aggregate Limit: {self.agg_limit:n}\n"
        if self.attachment:
            output += f"Attachment: {self.attachment:n}\n"
        output += f"Coverage: {self.coverage}\n"
        output += f"Notes: {self.notes}\n"
        return output
    
    def __post_init__(self):
        """Validate policy parameters after initialization."""
        if self.effective_date >= self.expiration_date:
            raise ValueError("Effective date must be before expiration date")
        if self.coinsurance is not None and not 0.0 <= self.coinsurance <= 1.0:
            raise ValueError("Coinsurance must be between 0.0 and 1.0")
        if self.retention_type not in ["deductible", "SIR"]:
            raise ValueError("Retention type must be 'deductible' or 'SIR'")
        # Validate non-negative values
        for field, value in [
            ("per_occ_retention", self.per_occ_retention),
            ("agg_retention", self.agg_retention),
            ("corridor_retention", self.corridor_retention),
            ("per_occ_limit", self.per_occ_limit),
            ("agg_limit", self.agg_limit),
            ("attachment", self.attachment),
            ("exposure_amount", self.exposure_amount)
        ]:
            if value is not None and value < 0:
                raise ValueError(f"{field} cannot be negative")
    
    def is_policy_active(self, evaluation_date: date) -> bool:
        """
        Check if the policy is active on a given date.
        
        Args:
            evaluation_date: Date to check policy status
            
        Returns:
            bool: True if policy is active on the evaluation date
            
        Examples:
            >>> from datetime import date
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31)
            ... )
            >>> terms.is_policy_active(date(2024, 6, 15))
            True
            >>> terms.is_policy_active(date(2025, 1, 1))
            False
        """
        return self.effective_date <= evaluation_date < self.expiration_date
    
    def apply_retention(self, loss_amount: np.ndarray | float, 
                       aggregate_losses_so_far: float = 0.0) -> tuple[np.ndarray | float, float]:
        """
        Apply retention (deductible or SIR) to loss amount.
        
        Args:
            loss_amount: Gross loss amount(s) before retention
            aggregate_losses_so_far: Total retained losses so far (for aggregate retention tracking)
            
        Returns:
            tuple: (net_loss_after_retention, retained_amount)
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     retention_type="deductible",
            ...     per_occ_retention=1000.0
            ... )
            >>> net_loss, retained = terms.apply_retention(2500.0)
            >>> net_loss
            1500.0
            >>> retained
            1000.0
        """
        # Convert to numpy array for consistent handling
        loss_array = np.atleast_1d(loss_amount)
        retained = np.zeros_like(loss_array)
        
        # Apply per-occurrence retention
        if self.per_occ_retention > 0:
            retained = np.minimum(loss_array, self.per_occ_retention)
            net_loss = loss_array - retained
        else:
            net_loss = loss_array.copy()
            
        # Apply aggregate retention if specified
        if self.agg_retention is not None and self.agg_retention > 0:
            # Calculate how much aggregate retention is remaining
            agg_retention_remaining = max(0, self.agg_retention - aggregate_losses_so_far)
            if agg_retention_remaining > 0:
                # Apply remaining aggregate retention
                total_retained = retained.sum()
                additional_retention = min(total_retained, agg_retention_remaining)
                # Proportionally reduce the retained amount if aggregate limit is reached
                if total_retained > 0:
                    retention_factor = additional_retention / total_retained
                    net_loss = net_loss + retained * (1 - retention_factor)
                    retained = retained * retention_factor
        
        # Return scalar if input was scalar
        if np.isscalar(loss_amount):
            return float(net_loss[0]), float(retained[0])
        return net_loss, retained
    
    def apply_limits(self, loss_amount: np.ndarray | float,
                    aggregate_paid_so_far: float = 0.0) -> tuple[np.ndarray | float, float]:
        """
        Apply per-occurrence and aggregate limits to loss amount.
        
        Args:
            loss_amount: Loss amount(s) after retention
            aggregate_paid_so_far: Total losses paid so far (for aggregate limit tracking)
            
        Returns:
            tuple: (limited_loss, amount_over_limit)
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     per_occ_limit=10000.0,
            ...     agg_limit=50000.0
            ... )
            >>> limited, excess = terms.apply_limits(15000.0)
            >>> limited
            10000.0
            >>> excess
            5000.0
        """
        # Convert to numpy array for consistent handling
        loss_array = np.atleast_1d(loss_amount)
        
        # Apply per-occurrence limit
        if self.per_occ_limit is not None:
            limited_loss = np.minimum(loss_array, self.per_occ_limit)
            amount_over_limit = loss_array - limited_loss
        else:
            limited_loss = loss_array.copy()
            amount_over_limit = np.zeros_like(loss_array)
            
        # Apply aggregate limit if specified
        if self.agg_limit is not None:
            # Calculate remaining aggregate limit
            agg_limit_remaining = max(0, self.agg_limit - aggregate_paid_so_far)
            
            # Apply aggregate limit
            cumsum = np.cumsum(limited_loss)
            within_agg_limit = cumsum <= agg_limit_remaining
            
            # Adjust losses that exceed aggregate limit
            if not within_agg_limit.all():
                # Find where we exceed the limit
                exceed_idx = np.where(~within_agg_limit)[0]
                if len(exceed_idx) > 0:
                    first_exceed = exceed_idx[0]
                    # Partial payment for the loss that crosses the limit
                    if first_exceed > 0:
                        prior_sum = cumsum[first_exceed - 1]
                    else:
                        prior_sum = 0
                    partial_payment = agg_limit_remaining - prior_sum
                    
                    # Update limited losses and excess
                    amount_over_limit[first_exceed] += limited_loss[first_exceed] - partial_payment
                    limited_loss[first_exceed] = partial_payment
                    
                    # Zero out all subsequent losses
                    amount_over_limit[first_exceed + 1:] += limited_loss[first_exceed + 1:]
                    limited_loss[first_exceed + 1:] = 0
        
        # Return scalar if input was scalar
        if np.isscalar(loss_amount):
            return float(limited_loss[0]), float(amount_over_limit[0])
        return limited_loss, amount_over_limit
    
    def apply_coinsurance(self, loss_amount: np.ndarray | float) -> np.ndarray | float:
        """
        Apply coinsurance to loss amount.
        
        Args:
            loss_amount: Loss amount(s) to apply coinsurance to
            
        Returns:
            Loss amount after coinsurance (insurer's portion)
            
        Notes:
            - coinsurance = 0.0 means insurer pays all (100%)
            - coinsurance = 1.0 means insured pays all (0% to insurer)
            - coinsurance = 0.2 means insured pays 20%, insurer pays 80%
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     coinsurance=0.2  # 20% coinsurance
            ... )
            >>> terms.apply_coinsurance(10000.0)
            8000.0  # Insurer pays 80%
        """
        if self.coinsurance is None:
            return loss_amount
            
        # Calculate insurer's share (1 - coinsurance)
        insurer_share = 1.0 - self.coinsurance
        
        if np.isscalar(loss_amount):
            return loss_amount * insurer_share
        return np.asarray(loss_amount) * insurer_share
    
    def apply_xol_attachment(self, ground_up_loss: np.ndarray | float) -> np.ndarray | float:
        """
        Apply excess-of-loss (XoL) attachment point.
        
        Args:
            ground_up_loss: Ground-up loss amount(s)
            
        Returns:
            Loss amount excess of attachment point
            
        Notes:
            XoL layers only respond to losses exceeding the attachment point.
            The attachment point is applied before any limits.
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     attachment=100000.0,
            ...     per_occ_limit=500000.0
            ... )
            >>> terms.apply_xol_attachment(150000.0)
            50000.0  # Only the excess over 100k
        """
        if self.attachment is None or self.attachment <= 0:
            return ground_up_loss
            
        if np.isscalar(ground_up_loss):
            return max(0, ground_up_loss - self.attachment)
            
        loss_array = np.asarray(ground_up_loss)
        return np.maximum(0, loss_array - self.attachment)
    
    def apply_policy_terms(self, loss_amount: float, 
                          aggregate_retained_so_far: float = 0.0,
                          aggregate_paid_so_far: float = 0.0) -> PolicyResult:
        """
        Apply all policy terms to a loss amount in the correct order.
        
        This is the main method that orchestrates the application of all policy
        features in the correct actuarial order.
        
        Args:
            loss_amount: Ground-up loss amount
            aggregate_retained_so_far: Total retained by insured so far (for agg retention)
            aggregate_paid_so_far: Total paid by insurer so far (for agg limits)
            
        Returns:
            PolicyResult: Detailed breakdown of the policy application
            
        Order of operations:
            1. Validate loss amount
            2. Apply retentions (deductible/SIR)
            3. Apply XoL attachment if present
            4. Apply limits (per-occurrence and aggregate)
            5. Apply corridor retention if present
            6. Apply coinsurance
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     retention_type="deductible",
            ...     per_occ_retention=1000.0,
            ...     per_occ_limit=10000.0,
            ...     coinsurance=0.2  # 20% coinsurance
            ... )
            >>> result = terms.apply_policy_terms(15000.0)
            >>> result.insurer_share  # (15000 - 1000) capped at 10000, then 80% of that
            8000.0
        """
        # 1. Validate loss amount
        if loss_amount < 0:
            raise ValueError("Loss amount cannot be negative")
            
        # 2. Apply retentions (deductible/SIR)
        loss_after_retention, retained = self.apply_retention(
            loss_amount, aggregate_retained_so_far)
        
        # 3. Apply XoL attachment if present
        excess_attachment = 0.0
        if self.attachment is not None and self.attachment > 0:
            # For XoL, we apply attachment to the gross loss
            loss_after_attachment = self.apply_xol_attachment(loss_amount)
            excess_attachment = loss_amount - loss_after_attachment
            
            # Adjust retention calculation for XoL
            if self.retention_type == "SIR":
                # SIR applies to losses excess of attachment
                loss_after_retention, retained = self.apply_retention(
                    loss_after_attachment, aggregate_retained_so_far)
            else:
                # For deductible on XoL, it applies after attachment
                loss_after_retention = loss_after_attachment
        else:
            loss_after_attachment = loss_amount
            
        # 4. Apply limits (per-occurrence and aggregate)
        limited_loss, amount_over_limit = self.apply_limits(
            loss_after_retention, aggregate_paid_so_far)
            
        # 5. Apply corridor retention if present
        corridor_retained = 0.0
        if self.corridor_retention is not None and self.corridor_retention > 0:
            corridor_retained = min(limited_loss, self.corridor_retention)
            limited_loss = limited_loss - corridor_retained
            
        # 6. Apply coinsurance
        insurer_share = self.apply_coinsurance(limited_loss)
        coinsurance_to_insured = limited_loss - insurer_share
        
        # Calculate total insured share
        insured_share = retained + corridor_retained + coinsurance_to_insured
        
        return PolicyResult(
            ground_up_loss=loss_amount,
            retained_amount=retained,
            loss_after_retention=loss_after_retention,
            excess_attachment=excess_attachment,
            loss_after_attachment=loss_after_attachment,
            limited_loss=limited_loss + corridor_retained,  # Before corridor
            amount_over_limit=amount_over_limit,
            corridor_retained=corridor_retained,
            insurer_share=insurer_share,
            insured_share=insured_share
        )
    
    def get_exposure_info(self) -> dict:
        """
        Get exposure information for rating and analysis.
        
        Returns:
            dict: Exposure base, amount, and related information
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     exposure_base=PAYROLL,
            ...     exposure_amount=1000000.0
            ... )
            >>> info = terms.get_exposure_info()
            >>> info['exposure_base_name']
            'Payroll'
        """
        return {
            'exposure_base': self.exposure_base,
            'exposure_base_name': self.exposure_base.name if self.exposure_base else None,
            'exposure_base_unit': self.exposure_base.unit if self.exposure_base else None,
            'exposure_amount': self.exposure_amount,
            'lob': self.lob,
            'lob_name': str(self.lob) if self.lob else None
        }
    
    def calculate_rate_per_unit(self, premium: float) -> float:
        """
        Calculate rate per unit of exposure.
        
        Args:
            premium: Total premium amount
            
        Returns:
            float: Rate per unit of exposure
            
        Examples:
            >>> terms = PolicyTerms(
            ...     effective_date=date(2024, 1, 1),
            ...     expiration_date=date(2024, 12, 31),
            ...     exposure_base=PAYROLL,
            ...     exposure_amount=1000000.0
            ... )
            >>> terms.calculate_rate_per_unit(5000.0)
            0.005  # $5 per $1000 of payroll
        """
        if self.exposure_amount <= 0:
            raise ValueError("Exposure amount must be positive to calculate rate")
        
        # For monetary exposures, rate is often per $1000 or $100
        if self.exposure_base and self.exposure_base.unit == "USD":
            return premium / (self.exposure_amount / 1000.0)
        else:
            # For other units, rate is per unit
            return premium / self.exposure_amount

# TODO: Add a Reinsurance policy dataclass.


@dataclass(slots=True)
class Inforce:
    """
    Bucket of policies sharing frequency and severity characteristics.

    Args:
        n_policies (int): Number of policies in this bucket.
        freq (FrequencyModel): Claim count distribution.
        sev (SeverityModel): Claim severity distribution.
        terms (PolicyTerms): Policy terms and layer definitions.
        name (str, optional): Bucket label. Defaults to "Unnamed Bucket".

    Methods:
        classical_sample(n_sims: int) -> np.ndarray: Monte Carlo aggregate loss sample.
    """
    n_policies:       int
    terms:            PolicyTerms
    frequency:        FrequencyModel
    severity:         SeverityModel
    name:             str = "Unnamed Bucket"

    def __add__(self, other):
        """
        Merge this bucket with another into a Portfolio.

        Args:
            other (Inforce): Another in-force bucket to merge.

        Returns:
            Inforce: New in-force bucket containing policies from both.

        Examples:
            >>> combined = bucket1 + bucket2
            >>> len(combined)
        """
        return Portfolio([self, other])

    def __len__(self):
        """
        Get the number of policies in this bucket.

        Returns:
            int: Number of policies in the bucket.

        Examples:
            >>> len(bucket)
        """
        return self.n_policies

    def __str__(self) -> str:
        output = f"Bucket: {self.name}\n" + \
            f"- Number of Policies: {self.n_policies:n}\n" + \
            f"- Frequency Model: {str(self.frequency)}\n" + \
            f"- Severity Model: {str(self.severity)}\n" + \
            f"- Policy Terms:\n"
        for pol_detail in str(self.terms).splitlines():
            output += f"  - {pol_detail}\n"
        return output

    @property
    def policies(self) -> int:
        """
        Get the number of policies in this bucket.

        Returns:
            int: Number of policies in the bucket.

        Examples:
            >>> bucket.policies
        """
        return self.n_policies

    def simulate(self, n_sims: int = 1) -> pd.Series | float:
        """
        Generate random variates from frequency and severity, returning aggregate losses.

        Args:
            n_sims (int): Number of simulations to run.

        Returns:
            pd.Series: Series containing simulated aggregates.

        Examples:
            >>> bucket.simulate(n_sims=1_000)
        """
        if n_sims is None or n_sims < 0 or not isinstance(n_sims, int):
            raise ValueError(
                "Number of simulations must be a positive integer.")
        if n_sims == 0:
            return 0.0
        
        # Wrap distributions for QMC if enabled
        if get_qmc_simulator() is not None:
            freq = wrap_for_qmc(self.frequency)
            sev = wrap_for_qmc(self.severity)
        else:
            freq = self.frequency
            sev = self.severity
            
        if n_sims == 1:
            # 1 simulation: one tuple of severity values, one per policy
            freq_samples = np.sum([freq.rvs()
                                   for _ in range(self.n_policies)])
            sim_result = np.sum(sev.rvs(int(freq_samples)))
            return sim_result
        else:
            # Multiple simulations: a tuple of tuples
            all_sims = []
            for _ in range(n_sims):
                freq_samples = np.sum([freq.rvs()
                                      for _ in range(self.n_policies)])
                if freq_samples == 0:
                    # No claims, no severity
                    all_sims.append(0)
                    continue
                sim_result = np.sum(sev.rvs(int(freq_samples)))
                all_sims.append(sim_result)
            return pd.Series(all_sims)


class Portfolio(list[Inforce]):
    """
    Portfolio of in-force buckets.

    Extends `list` to include portfolio-level operations.

    Examples:
        >>> p1 = Portfolio([bucket1])
        >>> p2 = Portfolio([bucket2])
        >>> combined = p1 + p2
    """

    def __init__(self,
                 buckets: list[Inforce] | Inforce,
                 backend: Optional[BackendManager] = None,
                 **kwargs):
        """
        Initialize the portfolio with a list of in-force buckets.

        Args:
            buckets (list[Inforce]): List of `Inforce` objects.
            backend (Optional[BackendManager]): Execution backend.

        Examples:
            >>> portfolio = Portfolio([gl_inforce, wc_inforce])
        """
        if isinstance(buckets, Inforce):
            # If a single Inforce is passed, convert it to a list
            buckets = [buckets]
        super().__init__(buckets)
        self.backend = backend if backend else set_backend(
            'quantum', **kwargs)
        # Local import to avoid circular dependency at module load
        from quactuary.pricing import PricingModel
        self.pricing_model = PricingModel(self)

    def __add__(self, other):
        """
        Merge this portfolio with another or add an in-force bucket.

        Args:
            other (Portfolio or Inforce): Another portfolio or an in-force bucket to merge.

        Returns:
            Portfolio: New portfolio containing buckets from both.

        Examples:
            >>> combined = portfolio1 + portfolio2
            >>> combined = portfolio + inforce_bucket
            >>> len(combined)
        """
        if isinstance(other, Portfolio):
            return Portfolio(list(self) + list(other))
        elif hasattr(other, 'n_policies'):
            return Portfolio(list(self) + [other])
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        """
        Merge this portfolio with another or add an in-force bucket in place.

        Args:
            other (Portfolio or Inforce): Another portfolio or an in-force bucket to merge.

        Returns:
            Portfolio: This portfolio with buckets from both.

        Examples:
            >>> portfolio += portfolio2
            >>> portfolio += inforce_bucket
            >>> len(portfolio)
        """
        if isinstance(other, Portfolio):
            self.extend(other)
        elif hasattr(other, 'n_policies'):
            self.append(other)
        else:
            raise NotImplementedError
        return self

    def __len__(self):
        """
        Get the number of buckets in the portfolio.

        Returns:
            int: Number of `Inforce` objects in the portfolio.

        Examples:
            >>> len(portfolio)
        """
        return self.policies

    def __str__(self) -> str:
        output = ""
        for bucket in self:
            output += str(bucket)
        return output

    @property
    def policies(self) -> int:
        """
        Compute the total number of policies in the portfolio.

        Returns:
            int: Sum of `n_policies` across all buckets.

        Examples:
            >>> portfolio.total_policies()
        """
        # Sum n_policies from each Inforce in the portfolio
        return sum(bucket.n_policies for bucket in self)

    def simulate(self, n_sims: int = 1) -> pd.Series | float:
        """
        Generate random variates from frequency and severity from all buckets in the portfolio, returning aggregate losses.

        Args:
            n_sims (int): Number of simulations to run.

        Returns:
            pd.Series: Series containing simulated aggregates.

        Examples:
            >>> portfolio.simulate(n_sims=1_000)
        """
        if n_sims is None or n_sims < 0 or not isinstance(n_sims, int):
            raise ValueError(
                "Number of simulations must be a positive integer.")
        if n_sims == 0:
            return 0.0
        bucket_sims = [bucket.simulate(n_sims) for bucket in self]
        if n_sims == 1:
            return sum(bucket_sims)
        else:
            grouped_by_sim = []
            for i in range(n_sims):
                sim_result = 0.0
                for bucket in bucket_sims:
                    if isinstance(bucket, pd.Series):
                        # pick the iᵗʰ simulation from this bucket
                        sim_result += bucket.iloc[i]
                    else:
                        # bucket is a scalar float
                        sim_result += bucket
                grouped_by_sim.append(sim_result)
            return pd.Series(grouped_by_sim)
