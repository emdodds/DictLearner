import numpy as np 
import tf_toposparse

class topo_test():

    def setup(self):
        pass

    def default_test(self):
        """Check that the default topology is correct. (Actually just check one row)"""
        topo = tf_toposparse.topology(shape=(25,25), sigma=np.sqrt(2))
        g = topo.get_matrix()
        row5 = np.array([1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0])
        assert np.allclose(row5, g[5]), g[5]

    def two_component_test(self):
        topo = tf_toposparse.topology(shape=(25,25), sigma=np.sqrt(2), ncomponents=2)
        g = topo.get_matrix()
        lilrow5 = [1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0]
        row5 = np.array(lilrow5 + [0]*25)
        assert np.allclose(row5, g[5]), g[5]
        row30 = np.array([0]*25 + lilrow5)
        assert np.allclose(row30, g[30]), g[30]

    def flat_sheet_test(self):
        topo = tf_toposparse.topology(shape=(25,25), sigma=np.sqrt(2), torus=False)
        g = topo.get_matrix()
        row5 = np.array([1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        assert np.allclose(row5, g[5]), g[5]

