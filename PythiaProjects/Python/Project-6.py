import pythia8
import fastjet
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------------
# CONFIG
# ----------------------------------
N_EVENTS = 12000
IMG_SIZE = 33          # 33x33 jet image
ETA_RANGE = 1.0        # Δη window
PHI_RANGE = 1.0        # Δφ window
JET_PT_MIN = 20.0

# ----------------------------------
# PYTHIA SETUP
# ----------------------------------
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 11")
pythia.readString("Beams:idB = -11")
pythia.readString("Beams:eCM = 91.")
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
pythia.init()

# ----------------------------------
# FASTJET SETUP
# ----------------------------------
jet_def = fastjet.JetDefinition(
    fastjet.antikt_algorithm, 0.6
)

# ----------------------------------
# JET IMAGE FUNCTION
# ----------------------------------
def make_jet_image(particles, jet):
    img = np.zeros((IMG_SIZE, IMG_SIZE))

    for p in particles:
        deta = p.eta() - jet.eta()
        dphi = math.atan2(
            math.sin(p.phi() - jet.phi()),
            math.cos(p.phi() - jet.phi())
        )

        if abs(deta) > ETA_RANGE or abs(dphi) > PHI_RANGE:
            continue

        x = int((deta + ETA_RANGE) / (2 * ETA_RANGE) * IMG_SIZE)
        y = int((dphi + PHI_RANGE) / (2 * PHI_RANGE) * IMG_SIZE)

        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            img[x, y] += p.pt()

    if img.sum() > 0:
        img /= img.sum()

    return img

# ----------------------------------
# DATASET GENERATION
# ----------------------------------
X = []
y = []

for _ in range(N_EVENTS):
    if not pythia.next():
        continue

    particles = []
    fj_particles = []

    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_info(p)
            fj_particles.append(pj)
            particles.append(p)

    if len(fj_particles) < 2:
        continue

    cs = fastjet.ClusterSequence(fj_particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=JET_PT_MIN)
    )

    if not jets:
        continue

    jet = jets[0]  # leading jet

    img = make_jet_image(particles, jet)

    label = 1 if jet.pt() > 40 else 0  # HARD vs SOFT jet

    X.append(img)
    y.append(label)

X = np.array(X)[..., np.newaxis]
y = np.array(y)

print("Jet images:", X.shape)

# ----------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------
# CNN MODEL
# ----------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu",
                  input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------
# TRAINING
# ----------------------------------
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------
# EVALUATION
# ----------------------------------
preds = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification report:\n")
print(classification_report(y_test, preds))

# ----------------------------------
# LIVE INFERENCE
# ----------------------------------
print("\nLive jet-image inference:\n")

for i in range(5):
    pythia.next()

    particles = []
    fj_particles = []

    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_info(p)
            fj_particles.append(pj)
            particles.append(p)

    cs = fastjet.ClusterSequence(fj_particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=JET_PT_MIN)
    )

    if not jets:
        continue

    jet = jets[0]
    img = make_jet_image(particles, jet)
    img = img[np.newaxis, ..., np.newaxis]

    pred = model.predict(img)[0][0]

    print(
        f"Jet pT={jet.pt():.1f} GeV →",
        "HARD" if pred > 0.5 else "SOFT"
    )

pythia.stat()
