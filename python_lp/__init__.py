# Local Projections Python Library
# Replication of Jordà & Taylor JEL Paper Methods

from .core import LocalProjections
from .smoothing import GaussianBasisFunction, smooth_irf
from .inference import (
    newey_west_se,
    lag_augmented_se,
    joint_significance_test,
    bonferroni_bands,
    bootstrap_bands
)
from .multipliers import FiscalMultiplier
from .gmm import GMM_LP, GMM_GBF

__version__ = "1.0.0"
__all__ = [
    'LocalProjections',
    'GaussianBasisFunction',
    'smooth_irf',
    'newey_west_se',
    'lag_augmented_se',
    'joint_significance_test',
    'bonferroni_bands',
    'bootstrap_bands',
    'FiscalMultiplier',
    'GMM_LP',
    'GMM_GBF'
]
