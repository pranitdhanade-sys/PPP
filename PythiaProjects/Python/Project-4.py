import pythia8
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. PYTHIA INITIALIZATION
# -------------------------------
pythia = pythia8.Pythia()

pythia.readString("Beams:idA = 11")
pythia.readString("Beams:idB = -11")
pythia.readString("Beams:eCM = 91.")
pythia.readString("WeakSingleBoson:ffbar2gmZ = on")

pythia.init()

# -------------------------------
# 2. DATA GENERATION
# -------------------------------
N_EVENTS = 20000

X = []
y = []

for _ in range(N_EVENTS):
    if not pythia.next():
        continue

    n_ch = 0
    sum_pt = 0.0
    max_pt = 0.0
    px_sum = 0.0
    py_sum = 0.0

    for p in pythia.event:
        if p.isFinal() and p.isCharged():
            pt = p.pT()
            n_ch += 1
            sum_pt += pt
            max_pt = max(max_pt, pt)
            px_sum += p.px()
            py_sum += p.py()

    if n_ch == 0:
        continue

    avg_pt = sum_pt / n_ch
    shape = math.sqrt(px_sum**2 + py_sum**2) / sum_pt

    # Label: hard vs soft event
    label = 1 if n_ch >= 20 else 0

    X.append([n_ch, sum_pt, avg_pt, max_pt, shape])
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Events generated:", len(X))

# -------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 4. MACHINE LEARNING MODEL
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 5. EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 6. LIVE EVENT CLASSIFICATION
# -------------------------------
print("\nLive classification:\n")

for i in range(10):
    pythia.next()

    n_ch = sum_pt = max_pt = 0.0
    px_sum = py_sum = 0.0

    for p in pythia.event:
        if p.isFinal() and p.isCharged():
            pt = p.pT()
            n_ch += 1
            sum_pt += pt
            max_pt = max(max_pt, pt)
            px_sum += p.px()
            py_sum += p.py()

    if n_ch == 0:
        continue

    avg_pt = sum_pt / n_ch
    shape = math.sqrt(px_sum**2 + py_sum**2) / sum_pt

    features = np.array([[n_ch, sum_pt, avg_pt, max_pt, shape]])
    prediction = model.predict(features)[0]

    print(
        f"Event {i:02d} | n_ch={n_ch:2d} |",
        "HARD" if prediction else "SOFT"
    )

# -------------------------------
# 7. PYTHIA STATISTICS
# -------------------------------
pythia.stat()
