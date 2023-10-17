import numpy as np
import dynamics as dyn
import utils_parameters as param

def test_dyn(xxt_init, inputs):

    """
    Return the states obtained by applying given inputs to the system in the whole time horizon

    Args
      - xxt_init \in \R^6 states at time 0
      - inputs \in \R^2 inputs from time 0 to T-2

    Return 
      - xxt states from time 0 to T-1
  
    """

    ns = param.ns
    TT = param.TT

    xxt = np.zeros((ns, TT))
    xxt[:,0] = xxt_init

    for tt in range(TT-1):

        xxt[:,tt+1] = dyn.dynamics(xxt[:,tt], inputs[:,tt])[0].squeeze()

    return xxt
