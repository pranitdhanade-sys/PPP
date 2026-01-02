import pythia8
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# 1️⃣ Detector smearing function
# -----------------------------
def smear_pt(pt, a=0.01, b=0.001):
    """Apply Gaussian smearing to muon pT"""
    sigma = pt * np.sqrt(a*a + (b*pt)**2)
    return np.random.normal(pt, sigma)

# -----------------------------
# 2️⃣ Initialize PYTHIA
# -----------------------------
pythia = pythia8.Pythia()

# Beams: proton-proton collisions at 13 TeV
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 13000.")

# Physics process: Z/γ* → μ⁺ μ⁻ only
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
pythia.readString("23:onMode = off")          # Turn off all Z decays
pythia.readString("23:onIfAny = 13")          # Only muons

pythia.init()

# -----------------------------
# 3️⃣ Event generation
# -----------------------------
n_events = 50000
generated = 0
selected = 0

z_mass_list = []
mu_pt_list = []

for _ in range(n_events):
    if not pythia.next():
        continue
    generated += 1

    # Collect final-state muons passing acceptance
    muons = []
    for p in pythia.event:
        if abs(p.id()) == 13 and p.isFinal():
            pt_smear = smear_pt(p.pT())
            if pt_smear > 20 and abs(p.eta()) < 2.4:
                muons.append((p, pt_smear))

    # Require exactly 2 opposite-charge muons
    if len(muons) != 2 or muons[0][0].id() * muons[1][0].id() > 0:
        continue

    # Reconstruct smeared four-momenta
    def smeared_vector(p, pt_new):
        scale = pt_new / p.pT()
        return p.p() * scale  # scales 4-momentum vector

    v1 = smeared_vector(muons[0][0], muons[0][1])
    v2 = smeared_vector(muons[1][0], muons[1][1])
    z = v1 + v2
    mass = z.m()

    # Apply Z mass window cut
    if 80 < mass < 100:
        selected += 1
        z_mass_list.append(mass)
        mu_pt_list.append(muons[0][1])
        mu_pt_list.append(muons[1][1])

pythia.stat()

# -----------------------------
# 4️⃣ Efficiency calculation
# -----------------------------
efficiency = selected / generated
print(f"Generated events: {generated}")
print(f"Selected events after cuts: {selected}")
print(f"Selection efficiency: {efficiency:.3f}")

# -----------------------------
# 5️⃣ Plot results
# -----------------------------
# Muon pT
plt.figure(figsize=(7,5))
plt.hist(mu_pt_list, bins=60, histtype='step', linewidth=2, color='blue')
plt.xlabel(r"$p_T^\mu$ [GeV]")
plt.ylabel("Events")
plt.title("Muon Transverse Momentum After Cuts")
plt.grid(True)
plt.savefig("mu_pt.png")
plt.show()

# Z mass
plt.figure(figsize=(7,5))
plt.hist(z_mass_list, bins=60, histtype='step', linewidth=2, color='red')
plt.xlabel(r"$M_{\mu\mu}$ [GeV]")
plt.ylabel("Events")
plt.title("Z Boson Mass After Selection Cuts")
plt.grid(True)
plt.savefig("z_mass.png")
plt.show()

# -----------------------------
# 6️⃣ Mass resolution
# -----------------------------
mu, sigma = norm.fit(z_mass_list)
print(f"Fitted Z mass peak: μ = {mu:.2f} GeV, σ = {sigma:.2f} GeV")
