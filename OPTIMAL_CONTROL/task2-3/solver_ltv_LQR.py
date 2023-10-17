import utils_parameters as param
import numpy as np

ns = param.ns
ni = param.ni

def ltv_LQR(AA, BB, QQ, RR, QQf, TT):

  """
	LQR for LTV system with (time-varying) cost	
	
  Args
    - AA (nn x nn (x TT)) matrix
    - BB (nn x mm (x TT)) matrix
    - QQ (nn x nn (x TT)), RR (mm x mm (x TT)) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """

  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
  
  # Evaluate KK
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

  return KK, PP
    