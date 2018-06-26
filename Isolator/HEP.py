import numpy as np

def dPhi(phi1, phi2):
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi: dPhi = 2*np.pi - dPhi
    return dPhi

def dEta(eta1, eta2):
    return abs(eta1-eta2)

def dR(phi1, eta1, phi2, eta2):
    return np.sqrt(dPhi(phi1, phi2)**2 + dEta(eta1, eta2)**2)
