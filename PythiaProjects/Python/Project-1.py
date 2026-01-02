import pythia8
import numpy as np

pythia = pythia8.Pythia()

# Beam setup
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 13000.")

# Physics process
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
pythia.readString("23:onMode = off")
pythia.readString("23:onIfAny = 13")

pythia.init()

mu_pt = []
mu_eta = []
z_mass = []

for i in range(20000):
    if not pythia.next():
        continue

    muons = []

    for p in pythia.event:
        if abs(p.id()) == 13 and p.isFinal():
            mu_pt.append(p.pT())
            mu_eta.append(p.eta())
            muons.append(p)

    if len(muons) == 2:
        z = muons[0].p() + muons[1].p()
        z_mass.append(z.m())

pythia.stat()

np.save("mu_pt.npy", mu_pt)
np.save("mu_eta.npy", mu_eta)
np.save("z_mass.npy", z_mass)
