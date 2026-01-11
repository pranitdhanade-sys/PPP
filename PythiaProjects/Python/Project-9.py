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
N_EVENTS    = 22000
MAX_PARTS   = 30
K           = 7
JET_PT_MIN  = 20.0

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
# PARTICLE CLOUD
# ----------------------------------
def make_cloud(particles, jet):
    cloud = []
    for p in particles:
        deta = p.eta() - jet.eta()
        dphi = math.atan2(
            math.sin(p.phi() - jet.phi()),
            math.cos(p.phi() - jet.phi())
        )
        cloud.append([p.pT(), deta, dphi, p.m()])

    cloud = sorted(cloud, key=lambda x: -x[0])
    cloud = cloud[:MAX_PARTS]

    while len(cloud) < MAX_PARTS:
        cloud.append([0, 0, 0, 0])

    return np.array(cloud, dtype=np.float32)

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
    cloud = make_cloud(particles, jet)

    label = 1 if jet.pt() > 40 else 0

    X.append(cloud)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset:", X.shape)

# ----------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------
# EDGE CONV (ParticleNet CORE)
# ----------------------------------
class EdgeConv(layers.Layer):
    def __init__(self, units, k):
        super().__init__()
        self.k = k
        self.mlp = models.Sequential([
            layers.Dense(units, activation="relu"),
            layers.Dense(units, activation="relu")
        ])

    def call(self, x):
        # x: (B, N, F)
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        # pairwise distance in (eta,phi)
        coords = x[:, :, 1:3]
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        dist = tf.reduce_sum((a - b)**2, axis=-1)

        _, idx = tf.math.top_k(-dist, k=self.k+1)
        idx = idx[:, :, 1:]

        neighbors = tf.gather(x, idx, batch_dims=1)
        xi = tf.expand_dims(x, 2)
        edge_feat = tf.concat([xi, neighbors - xi], axis=-1)

        h = self.mlp(edge_feat)
        return tf.reduce_max(h, axis=2)

# ----------------------------------
# PARTICLENET MODEL
# ----------------------------------
inputs = layers.Input(shape=(MAX_PARTS, 4))

x = EdgeConv(64, K)(inputs)
x = EdgeConv(128, K)(x)
x = EdgeConv(256, K)(x)

x = layers.GlobalAveragePooling1D()(x)
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
    X_tr, y_tr,
    epochs=18,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------
# EVALUATION
# ----------------------------------
preds = (model.predict(X_te) > 0.5).astype(int)
print("\nClassification Report:\n")
print(classification_report(y_te, preds))

# ----------------------------------
# LIVE INFERENCE
# ----------------------------------
print("\nLive ParticleNet inference:\n")

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
    cloud = make_cloud(particles, jet)
    prob = model.predict(cloud[np.newaxis, ...])[0][0]

    print(
        f"Jet pT={jet.pt():.1f} GeV →",
        "HARD" if prob > 0.5 else "SOFT",
        f"(p={prob:.2f})"
    )

pythia.stat()
