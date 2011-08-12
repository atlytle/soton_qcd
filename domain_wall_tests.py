import unittest
import domain_wall as dw
import numpy as np

class TestGammaTraces(unittest.TestCase):
    
    def test_traces(self):
        "Ensure Tr(hc(G[i]).G[j]) = 4 delta_{ij}"
        traces = np.array([[np.trace(np.dot(dw.hc(dw.G[i]), dw.G[j]))/4 
                            for i in range(16)] for j in range(16)])
        np.testing.assert_array_equal(traces, np.identity(16, dtype=complex))

if __name__ == "__main__":
    unittest.main()
                        
