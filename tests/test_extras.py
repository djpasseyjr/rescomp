import unittest
import rescomp as rc
import numpy as np

class TestExtras(unittest.TestCase):

    def test_accdur(self):
        pass

    def test_relerr(self):
        x = np.array([3, 4])
        y = np.array([-2, -8])
        assert rc.relerr(x, y, order=2) == 13/5
        assert rc.relerr(x, y, order='inf') == 12/4
        X = np.array([[2, 2], [2, 2]])
        Y = np.array([[1, 0.5], [1.5, 2]])
        assert np.all(rc.relerr(X, Y, order=2, axis=0) == np.array([1.25**.5 / 8**.5, 1.5 / 8**.5]))
        assert np.all(rc.relerr(X.T, Y.T, order="inf", axis=1) == np.array([1/2, 1.5/2]))
        
    def test_sys_fit(self):
        pass

    def test_train_test(self):
        pass
    
    def test_nrmse(self):
        # Compare with results from R (hydroGOF package)
        x = np.array([2,3,4,5])
        y = np.array([2,3,5,5])
        assert np.isclose(rc.nrmse(x,y), 0.387298)
        A = np.array([[2,3],[4,5]])
        B = np.array([[2,3],[5,5]])
        assert np.all(np.isclose(rc.nrmse(A,B, axis=1), np.array([1, 0])))
        assert np.all(np.isclose(rc.nrmse(A,B, axis=0), np.array([0, .5])))
        
    def test_valid_prediction_index(self):
        err = [0,0,0,0,0,0,0,1,0,0]
        idx = rc.valid_prediction_index(err, .5)
        assert idx == 7
        
