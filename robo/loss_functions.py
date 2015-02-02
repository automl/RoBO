#import math
import numpy as np

def logLoss(logP, lmb, lPred, *args):
    try:
        H =   - np.sum(np.multiply(np.exp(logP), (logP + lmb))) # current entropy
    
        dHp = - np.sum(np.multiply(np.exp(lPred), np.add(lPred, lmb)), axis=0) - H # @minus? If you change it, change it above in H, too!
    except:
        print "H = ", H, "logP = ",  logP, "lmb = ", lmb
    return dHp