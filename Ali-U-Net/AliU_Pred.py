import numpy as np
import tensorflow as tf
import os
import argparse
import time

# Time tracking
start = time.time()

# Thread control
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Recoding nucleotides to integers
def recode_seq(seq):
    a = np.empty(len(seq), dtype=np.uint8)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    for i, c in enumerate(seq):
        a[i] = mapping.get(c.upper(), 4)  # default to gap if unrecognized
    return a

# Decode model predictions
nucleotide = ["A", "C", "G", "T", "-"]
def make_predict_sequences(pred_array):
    sequences = []
    for sample in pred_array:
        sequence = []
        for position in sample:
            index = np.argmax(position)
            sequence.append(nucleotide[index])
        sequences.append(''.join(sequence))
    return sequences

# Argument parsing
parser = argparse.ArgumentParser(description="Predict alignments from an .npz file.")
parser.add_argument("rows", type=int, help="Number of rows (sequences).")
parser.add_argument("columns", type=int, help="Number of columns (sequence length).")
parser.add_argument("model", type=str, help="Trained model file (HDF5).")
parser.add_argument("npz_file", type=str, help="Path to .npz file with 'x' key.")
parser.add_argument("output_prefix", type=str, help="Prefix for output FASTA files.")
args = parser.parse_args()

# Load model and input data
model = tf.keras.models.load_model(args.model)
data = np.load(args.npz_file)

x_encoded = data['x']  # shape: (N, rows, columns)
y_encoded = data['y']

# Predict
predictions = model.predict(x_encoded, verbose=1)

for idx in range(5):
    # 1) Decode the raw “shifted” input back to letters
    x_raw_int = np.argmax(x_encoded[idx], axis=-1)       # shape (rows,columns)
    x_raw_seqs = np.array([
      ''.join(nucleotide[i] for i in row) 
      for row in x_raw_int
    ])

    # 2) Decode the ground-truth alignment
    y_true_int = np.argmax(y_encoded[idx], axis=-1)
    y_true_seqs = np.array([
      ''.join(nucleotide[i] for i in row) 
      for row in y_true_int
    ])

    # 3) Run prediction
    y_pred_probs = model.predict(x_encoded[idx:idx+1])   # shape (1, rows, columns, 5)
    y_pred_seqs = make_predict_sequences(y_pred_probs[0])

    # 4) Print them side by side, highlighting mismatches
    for i, (raw, true, pred) in enumerate(zip(x_raw_seqs, y_true_seqs, y_pred_seqs)):
      print(f"Row {i+1}:")
      print("  Input : ", raw)
      print("  Ground: ", true)
      print("  Pred  : ", pred)
      # Mark mismatches with a simple caret under the offending chars:
      diff_line = ''.join('^' if true[j] != pred[j] else ' ' for j in range(len(true)))
      print("           ", diff_line)
      print()

    # 5) (Optional) Compute overall character-level accuracy for this sample:
    total = prod = 0
    for t, p in zip(y_true_seqs, y_pred_seqs):
      for tc, pc in zip(t, p):
        total += 1
        prod += (tc == pc)
    print(f"Sample {idx} accuracy: {prod/total:.2%}")

# Write each prediction to a separate FASTA file
for idx, pred in enumerate(predictions):
    aligned_seqs = make_predict_sequences(pred)
    output_file = f"{args.output_prefix}_{idx+1}.fasta"
    with open(output_file, "w") as f:
        for seq_idx, seq in enumerate(aligned_seqs):
            f.write(f">Seq{seq_idx+1}\n{seq}\n")

end = time.time()
print(f"Total prediction time: {end - start:.2f} seconds")
