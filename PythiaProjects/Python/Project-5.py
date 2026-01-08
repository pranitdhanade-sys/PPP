import pythia8
import fastjet
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# 1. PYTHIA INITIALIZATION
# ---------------------------------
pythia = pythia8.Pythia()

pythia.readString("Beams:idA = 11")
pythia.readString("Beams:idB = -11")
pythia.readString("Beams:eCM = 91.")
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")

pythia.init()

# ---------------------------------
# 2. FASTJET SETUP
# ---------------------------------
R = 0.6
jet_def = fastjet.JetDefinition(
    fastjet.antikt_algorithm, R
)

# ---------------------------------
# 3. DATA GENERATION
# ---------------------------------
N_EVENTS = 15000

X = []
y = []

for _ in range(N_EVENTS):
    if not pythia.next():
        continue

    particles = []

    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            particles.append(
                fastjet.PseudoJet(
                    p.px(), p.py(), p.pz(), p.e()
                )
            )

    if len(particles) < 2:
        continue

    cs = fastjet.ClusterSequence(particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=5.0)
    )

    n_jets = len(jets)
    if n_jets == 0:
        continue

    jet_pts = [j.pt() for j in jets]
    jet_etas = [abs(j.eta()) for j in jets]

    # Jet-level features
    max_jet_pt = max(jet_pts)
    avg_jet_pt = sum(jet_pts) / n_jets
    sum_jet_pt = sum(jet_pts)
    avg_eta = sum(jet_etas) / n_jets

    # Physics-inspired label
    label = 1 if (n_jets >= 3 and max_jet_pt > 15) else 0

    X.append([
        n_jets,
        max_jet_pt,
        avg_jet_pt,
        sum_jet_pt,
        avg_eta
    ])
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total jet-events:", len(X))

# ---------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# 5. MACHINE LEARNING MODEL
# ---------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------
# 6. EVALUATION
# ---------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------------
# 7. LIVE JET-BASED CLASSIFICATION
# ---------------------------------
print("\nLive jet-based classification:\n")

for i in range(10):
    pythia.next()

    particles = []
    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            particles.append(
                fastjet.PseudoJet(
                    p.px(), p.py(), p.pz(), p.e()
                )
            )

    if len(particles) < 2:
        continue

    cs = fastjet.ClusterSequence(particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=5.0)
    )

    if len(jets) == 0:
        continue

    jet_pts = [j.pt() for j in jets]
    jet_etas = [abs(j.eta()) for j in jets]

    features = np.array([[
        len(jets),
        max(jet_pts),
        sum(jet_pts) / len(jets),
        sum(jet_pts),
        sum(jet_etas) / len(jets)
    ]])

    pred = model.predict(features)[0]

    print(
        f"Event {i:02d} | jets={len(jets)} | "
        f"max pT={max(jet_pts):.1f} →",
        "HARD" if pred else "SOFT"
    )

# ---------------------------------
# 8. PYTHIA STATISTICS
# ---------------------------------
pythia.stat()
