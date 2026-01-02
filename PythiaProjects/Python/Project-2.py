import pythia8
import numpy as np

pythia = pythia8.Pythia()

pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 13000.")
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
pythia.readString("23:onMode = off")
pythia.readString("23:onIfAny = 13")

pythia.init()

z_mass = []
mu_pt = []

for _ in range(50000):
    if not pythia.next():
        continue

    muons = []

    for p in pythia.event:
        if abs(p.id()) == 13 and p.isFinal():
            if p.pT() > 20 and abs(p.eta()) < 2.4:
                muons.append(p)

    if len(muons) != 2:
        continue

    if muons[0].id() * muons[1].id() > 0:
        continue

    z = muons[0].p() + muons[1].p()
    mass = z.m()

    if 80 < mass < 100:
        z_mass.append(mass)
        mu_pt.append(muons[0].pT())
        mu_pt.append(muons[1].pT())

pythia.stat()

np.save("z_mass_cut.npy", z_mass)
np.save("mu_pt_cut.npy", mu_pt)
