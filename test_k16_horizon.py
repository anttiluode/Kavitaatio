"""Regression checks for K16 causal-interaction experiments."""
import unittest
import numpy as np
import experiment
from k16_horizon import setup,wave_trace,inclusion_exclusion
from waypoint_tour import tour


class TestK16(unittest.TestCase):
    def test_k8_solver_matches_original(self):
        mask,sectors,src,receiver=setup(8)
        geometry=experiment.geometry()
        np.testing.assert_array_equal(mask,geometry[0])
        np.testing.assert_array_equal(sectors,geometry[1])
        bits=np.zeros(8,dtype=bool)
        bits[[0,3,7]]=True
        expected,_=experiment.simulate(bits,0)
        observed=wave_trace(mask,sectors,src,receiver,bits,.28,350)[145:345:2]
        np.testing.assert_allclose(expected[:,0],observed,atol=3e-14,rtol=0)

    def test_all_16_sectors_present(self):
        mask,sectors,_,_=setup(16)
        self.assertEqual(set(np.unique(sectors[mask])),set(range(16)))

    def test_geometric_tour_precedes_measured_onset(self):
        for K,subset in ((8,(0,2,3,4,6,7)),(16,tuple(range(5)))):
            lower=tour(K,subset)['earliest_arrival_lower_bound_steps']
            measured,_=inclusion_exclusion(K,subset,nsteps=440)
            self.assertGreaterEqual(measured['onset_at_0.0001'],lower-2.)

    def test_global_phase_does_not_create_new_sensor_information(self):
        mask,sectors,src,receiver=setup(16)
        no=np.zeros(16,dtype=bool)
        base=wave_trace(mask,sectors,src,receiver,no,.28,350)[145:345:2]
        one=[]
        for k in range(16):
            bits=no.copy()
            bits[k]=True
            one.append(wave_trace(mask,sectors,src,receiver,bits,.28,350)[145:345:2]-base)
        matrix=np.array(one).T
        rank=np.linalg.matrix_rank(matrix)
        rotated=np.concatenate((matrix,-matrix,1j*matrix,-1j*matrix))
        self.assertEqual(np.linalg.matrix_rank(rotated),rank)


if __name__=='__main__':
    unittest.main()
