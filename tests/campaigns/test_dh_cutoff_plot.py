"""Paired comparisons retain cross-scheme correlations in their uncertainty."""
import unittest
import numpy as np
from tools.campaigns.dh_cutoff_plot import paired_gain_uncertainty


class PairedGainTests(unittest.TestCase):
    def test_identical_and_scaled_estimators_have_zero_ratio_uncertainty(self):
        rng=np.random.default_rng(751)
        ref=rng.normal(size=(18,3,7));old=rng.normal(size=(18,3,7))
        data=np.stack((ref,old,2.5*old+4,old),axis=-1).astype(complex)
        error=paired_gain_uncertainty(data,0,[2,3],[1,1])
        np.testing.assert_allclose(error,0,atol=6e-15)
        # These schemes separately have nonzero SE. Independent error propagation
        # would miss the exact cancellation of their shared cfg fluctuations.
        self.assertTrue(np.all(data.real.std(0,ddof=1)>0))


if __name__=='__main__':unittest.main()
