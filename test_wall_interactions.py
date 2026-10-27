"""Fast regression controls; no 256-state cache or plotting dependencies."""
import unittest
import numpy as np
import experiment as exp
from wall_interactions import fit,BITS,ORDER


class TestWallInteractions(unittest.TestCase):
    def test_default_material_strength_unchanged(self):
        bits=BITS[17]
        old,_=exp.simulate(bits,0)
        new,_=exp.simulate(bits,0,stiffness=.28)
        np.testing.assert_array_equal(old,new)

    def test_pair_interaction_and_exact_reconstruction(self):
        setup=exp.geometry()
        off=np.zeros(8)
        a=off.copy();b=off.copy()
        a[0]=1;b[7]=1
        y0,_=exp.simulate(off,0,setup)
        ya,_=exp.simulate(a,0,setup)
        yb,_=exp.simulate(b,0,setup)
        yab,_=exp.simulate(a+b,0,setup)
        mixed=yab-ya-yb+y0
        self.assertGreater(np.linalg.norm(mixed),1e-5)
        np.testing.assert_allclose(y0+(ya-y0)+(yb-y0)+mixed,yab,atol=1e-12)

    def test_no_holdout_leakage(self):
        rng=np.random.default_rng(456)
        data=rng.normal(size=(256,3,10))
        for degree in (1,2,3,4):
            corrupted=data.copy()
            corrupted[ORDER>degree]=np.nan
            p,_=fit(data,degree)
            q,_=fit(corrupted,degree)
            np.testing.assert_array_equal(p,q)
            np.testing.assert_allclose(p[ORDER<=degree],data[ORDER<=degree],atol=1e-12)

    def test_exact_cubic_synthetic(self):
        rng=np.random.default_rng(123)
        linear=rng.normal(size=(8,3,5))
        triple=rng.normal(size=(8,8,8,3,5))*.03
        data=np.empty((256,3,5))
        for mask,bits in enumerate(BITS):
            v=np.einsum('i,ijk->jk',bits,linear)
            for i in range(8):
                for j in range(i+1,8):
                    for k in range(j+1,8):
                        v+=bits[i]*bits[j]*bits[k]*triple[i,j,k]
            data[mask]=v
        p,_=fit(data,3)
        np.testing.assert_allclose(p,data,atol=1e-12)
        p2,_=fit(data,2)
        self.assertGreater(np.max(np.abs(p2-data)),.01)


if __name__=='__main__':unittest.main()
