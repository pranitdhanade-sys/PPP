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
N_EVENTS   = 15000
IMG_SIZE   = 33
ETA_RANGE  = 1.0
PHI_RANGE  = 1.0
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
# JET IMAGE BUILDER
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
# DATA GENERATION
# ----------------------------------
X, y = [], []

for _ in range(N_EVENTS):
    if not pythia.next():
        continue

    fj_particles = []
    particles = []

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
    img = make_jet_image(particles, jet)

    label = 1 if jet.pt() > 40 else 0
    X.append(img)
    y.append(label)

X = np.array(X)[..., np.newaxis]
y = np.array(y)

print("Dataset:", X.shape)

# ----------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------
# RESNET BLOCK
# ----------------------------------
def res_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

# ----------------------------------
# RESNET MODEL
# ----------------------------------
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = res_block(x, 32)
x = layers.MaxPooling2D(2)(x)

x = res_block(x, 64)
x = layers.MaxPooling2D(2)(x)

x = res_block(x, 128)

x = layers.GlobalAveragePooling2D()(x)
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
    epochs=12,
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
print("\nLive ResNet jet inference:\n")

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
    img = make_jet_image(particles, jet)
    img = img[np.newaxis, ..., np.newaxis]

    prob = model.predict(img)[0][0]

    print(
        f"Jet pT={jet.pt():.1f} GeV →",
        "HARD" if prob > 0.5 else "SOFT",
        f"(p={prob:.2f})"
    )

pythia.stat()
