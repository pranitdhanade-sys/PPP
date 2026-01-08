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
N_EVENTS     = 18000
MAX_PARTS    = 30        # particles per jet
JET_PT_MIN   = 20.0

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
# BUILD PARTICLE CLOUD
# ----------------------------------
def make_particle_cloud(particles, jet):
    cloud = []

    for p in particles:
        deta = p.eta() - jet.eta()
        dphi = math.atan2(
            math.sin(p.phi() - jet.phi()),
            math.cos(p.phi() - jet.phi())
        )

        cloud.append([
            p.pT(),
            deta,
            dphi,
            p.m()
        ])

    cloud = sorted(cloud, key=lambda x: -x[0])
    cloud = cloud[:MAX_PARTS]

    while len(cloud) < MAX_PARTS:
        cloud.append([0, 0, 0, 0])

    return np.array(cloud)

# ----------------------------------
# DATASET GENERATION
# ----------------------------------
X, y = [], []

for _ in range(N_EVENTS):
    if not pythia.next():
        continue

    fj_particles, particles = [], []

    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            fj_particles.append(
                fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            )
            particles.append(p)

    if len(fj_particles) < 2:
        continue

    cs = fastjet.ClusterSequence(fj_particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=JET_PT_MIN)
    )

    if not jets:
        continue

    jet = jets[0]

    cloud = make_particle_cloud(particles, jet)

    label = 1 if jet.pt() > 40 else 0

    X.append(cloud)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Particle clouds:", X.shape)

# ----------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------
# POINTNET MODEL
# ----------------------------------
inputs = layers.Input(shape=(MAX_PARTS, 4))

x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)

x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

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
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------
# EVALUATION
# ----------------------------------
preds = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ----------------------------------
# LIVE INFERENCE
# ----------------------------------
print("\nLive PointNet jet inference:\n")

for i in range(5):
    pythia.next()

    fj_particles, particles = [], []
    for p in pythia.event:
        if p.isFinal() and p.isVisible():
            fj_particles.append(
                fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            )
            particles.append(p)

    cs = fastjet.ClusterSequence(fj_particles, jet_def)
    jets = fastjet.sorted_by_pt(
        cs.inclusive_jets(ptmin=JET_PT_MIN)
    )

    if not jets:
        continue

    jet = jets[0]
    cloud = make_particle_cloud(particles, jet)
    cloud = cloud[np.newaxis, ...]

    prob = model.predict(cloud)[0][0]

    print(
        f"Jet pT={jet.pt():.1f} GeV →",
        "HARD" if prob > 0.5 else "SOFT",
        f"(p={prob:.2f})"
    )

pythia.stat()
