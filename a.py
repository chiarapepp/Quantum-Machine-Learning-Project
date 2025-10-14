import numpy as np
import pandas as pd
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------
# 1) Data loading & preprocessing (placeholder)
# ---------------------------
def load_and_preprocess_nf_unsw(csv_path, selected_features):
    """
    Placeholder: adapt to the NF-UNSW-NB15 csv structure.
    - selected_features: list of column names to use
    Returns: X: DataFrame[n_samples, n_features], y: ndarray[n_samples]
    """
    df = pd.read_csv(csv_path)  # adapt if dataset split into multiple files
    df = df.dropna(subset=selected_features + ['label'])  # ensure no missing
    X = df[selected_features].copy()
    # Convert labels to binary: paper focuses on binary classification (attack vs benign)
    y = (df['label'] != 'BENIGN').astype(int).to_numpy()  # adjust depending on dataset labels
    # Simple normalization (min-max) — paper quantizes later
    X = (X - X.min())/(X.max() - X.min())
    return X, y

# ---------------------------
# 2) Encoding: classical -> quantum angle quantization in [0, pi]
# ---------------------------
def quantize_to_angle(value, bins=720):  # default granularity: 0.25 degrees -> 360/0.25 = 1440 but using 720 as example
    """
    Map a normalized value in [0,1] to angle in [0, pi].
    bins param controls quantization granularity (paper mentions 0.25° granularity as min).
    """
    # clip to 0..1 then quantize
    v = float(np.clip(value, 0.0, 1.0))
    idx = int(np.floor(v * (bins - 1)))
    angle = idx / (bins - 1) * np.pi
    return angle


import numpy as np
from qiskit import QuantumCircuit

def quantized_angle_encoding(sample, feature_bins=None, min_angle=0, max_angle=np.pi, granularity_deg=0.25):
    """
    Esegue la codifica di un singolo campione in angoli quantizzati [0, π].
    
    Args:
        sample (np.array): vettore delle feature (es. [0.3, 0.8, 0.1])
        feature_bins (list or None): lista di array di bin edges per ogni feature (se None → normalizzazione)
        min_angle (float): angolo minimo (default 0)
        max_angle (float): angolo massimo (default π)
        granularity_deg (float): granularità minima in gradi (default 0.25°)
    Returns:
        np.array: angoli θ quantizzati per ciascuna feature
    """
    n_features = len(sample)
    quantized_angles = np.zeros(n_features)
    step = np.deg2rad(granularity_deg)  # passo di quantizzazione in radianti

    for j in range(n_features):
        x = sample[j]

        if feature_bins is not None:
            # Se abbiamo bin predefiniti (percentili o categorie)
            bins = feature_bins[j]
            bin_index = np.digitize(x, bins) - 1
            bin_index = np.clip(bin_index, 0, len(bins) - 1)
            theta = min_angle + (max_angle - min_angle) * (bin_index / len(bins))
        else:
            # Normalizzazione semplice
            theta = (x / np.max(sample)) * max_angle

        # Quantizzazione dell'angolo
        theta = np.round(theta / step) * step
        quantized_angles[j] = theta

    return quantized_angles


def dataframe_to_circuit_list(X_df, bins=720):
    """
    Given a DataFrame X_df (n_samples x n_features), return list of cirq.Circuit
    where each feature corresponds to a qubit with a single Ry(angle) (or RY) rotation.
    """
    n_features = X_df.shape[1]
    qubits = [cirq.GridQubit(0, i) for i in range(n_features)]
    circuits = []
    for _, row in X_df.iterrows():
        circuit = cirq.Circuit()
        for i, val in enumerate(row):
            angle = quantize_to_angle(val, bins=bins)
            circuit.append(cirq.ry(angle)(qubits[i]))
        circuits.append(circuit)
    return circuits, qubits

# ---------------------------
# 3) Build parametrized QNN model (TFQ + Keras)
# ---------------------------
def make_qnn_circuit(qubits, n_layers=3, entangler='cz'):
    """
    Construct a parametrized variational circuit.
    n_layers: number of variational layers (paper experimented with 6/8 but uses lean)
    entangler: 'cz' or 'cnot' or 'iswap' - simple entanglement pattern
    Returns (model_circuit, symbols)
    """
    circuit = cirq.Circuit()
    symbols = []
    n_qubits = len(qubits)

    for layer in range(n_layers):
        # single-qubit parameterized rotations
        for i, q in enumerate(qubits):
            sym = sympy.symbols(f'theta_{layer}_{i}')
            circuit.append(cirq.rx(sym)(q))
            symbols.append(sym)
        # add an entangling layer (ring)
        for i in range(n_qubits - 1):
            if entangler == 'cz':
                circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
            elif entangler == 'cnot':
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            else:
                circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
        # close the ring
        if n_qubits > 1:
            circuit.append(cirq.CZ(qubits[-1], qubits[0]))

    return circuit, symbols

def build_tfq_model(qubits, readout_qubit_index=0, n_layers=3):
    """Create a TFQ Keras model: input = circuits, output = scalar prob (sigmoid)"""
    # prepare parameterized circuit that will be appended to each input state
    v_circuit, symbols = make_qnn_circuit(qubits, n_layers=n_layers)
    # readout operator on one qubit (Z expectation -> map to prob)
    readout = cirq.Z(qubits[readout_qubit_index])

    # Inputs
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='circuit_input')
    # PQC layer: parameterized quantum circuit producing expectation of readout
    pqc = tfq.layers.PQC(v_circuit, readout)(circuit_input)
    # pqc returns expectation in [-1,1] — map to [0,1] via (x+1)/2, then use as probability
    prob = tf.keras.layers.Lambda(lambda x: (x + 1.0) / 2.0)(pqc)
    # optionally add a small dense layer if you want classical post-processing
    out = tf.keras.layers.Dense(1, activation='sigmoid')(prob)
    model = tf.keras.Model(inputs=circuit_input, outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

# ---------------------------
# 4) Helpers: convert circuits & labels to TFQ tensors
# ---------------------------
def circuits_to_tensor(circuits):
    return tfq.convert_to_tensor(circuits)

# ---------------------------
# 5) Training & evaluation
# ---------------------------
def train_and_evaluate(X, y, n_layers=3, batch_size=32, epochs=10):
    # convert DataFrame -> circuits
    circuits, qubits = dataframe_to_circuit_list(X)
    x_tensor = circuits_to_tensor(circuits)
    y_np = np.array(y).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(x_tensor, y_np, test_size=0.2, random_state=42)
    # small val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = build_tfq_model(qubits, readout_qubit_index=0, n_layers=n_layers)
    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

    # Predict (probabilities)
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Test F1: {f1:.4f} precision: {prec:.4f} recall: {rec:.4f}")

    # compute certainty factor as described in paper (paper's exact formula to be implemented)
    # Here: a proxy — certainty = |prob - 0.5| scaled to [0,1]
    certainty = np.abs(y_prob - 0.5) * 2.0
    return {'f1': f1, 'precision': prec, 'recall': rec, 'certainty_mean': certainty.mean()}

# ---------------------------
# 6) Example run (replace csv_path and selected features)
# ---------------------------
if __name__ == "__main__":
    CSV_PATH = "UNSW_NB15.csv"  # replace with actual path
    # Example feature list — adapt to the paper's chosen features (paper provides a table)
    selected_features = ['sbytes', 'sttl', 'dttl', 'stcpb', 'dtcpb']  # REPLACE WITH actual features from paper
    X, y = load_and_preprocess_nf_unsw(CSV_PATH, selected_features)
    results = train_and_evaluate(X, y, n_layers=3, batch_size=16, epochs=5)
    print("Results:", results)