import unittest
import rescomp as rc

class TestExtras(unittest.TestCase):

    def test_accdur():
        pass

    def test_relerr():
        x = np.array([3, 4])
        y = np.array([-2, -8])
        assert rc.relerr(x, y, order=2) == 13/5
        assert rc.relerr(x, y, order='inf') == 12/4
        X = np.array([[2, 2], [2, 2]])
        Y = np.array([[1, 0.5], [1.5, 2]])
        assert np.all(rc.relerr(X, Y, order=2, axis=0) == np.array([1.25**.5 / 8**.5, 1.5 / 8**.5]))
        assert np.all(rc.relerr(X.T, Y.T, order="inf", axis=1) == np.array([1/2, 1.5/2]))
        
    def test_sys_fit():
        pass

    def test_train_test():
        pass
