import argparse
import random
import numpy as np
import tensorflow as tf
import os
import sys



parser = argparse.ArgumentParser(description="Simulate a study case without internal gaps.")

parser.add_argument("rows", type=int, help="The number of rows.")
parser.add_argument("columns", type=int, help="The number of columns.")
parser.add_argument("margin", type=int, help="The margin size.")
parser.add_argument("skip_rows", type=int, help="The number of rows to be left without gaps.")
parser.add_argument("alignments", type=int, help="The number of alignments.")
parser.add_argument("filename", type=str, help="The file name.")

args = parser.parse_args()

rows = args.rows
columns = args.columns
margin = args.margin
skip_rows = args.skip_rows
arrays = args.alignments
filename = args.filename

def random_numbers_adding_up_100():
  while True:
      r1 = random.randint(10,100)
      r2 = random.randint(10,100)
      r3 = random.randint(10,100)
      r4 = random.randint(10,100)

      s = (r1+r2+r3+r4)

      r1 = r1/s
      r2 = r2/s
      r3 = r3/s
      r4 = r4/s

      yield (r1, r2, r3, r4)

gen = random_numbers_adding_up_100()

def DNA_profile(columns):
  probabilities = [1.00,0.00,0.00,0.00]
  profile=[]
  for i in range(columns):
    prob = random.sample(probabilities, len(probabilities)) #Shuffle around so that the spotlight is always on a different nucleotide.
    profile+=[prob]
  return profile

nucleotides = ["A", "C", "G", "T"]

def make_sequences(rows,columns):
  profile = DNA_profile(columns)
  sequences = []
  for i in range(rows):
    sequence = []
    for i in range(columns):
      sequence += random.choices(nucleotides, weights=profile[i], k=1)
    sequence = ''.join(sequence)
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences


sequences = make_sequences(8,32)

def recode_seq(seq): #This one is for recoding everything.
  seq_len = len(seq)
  a = np.empty(shape=(seq_len), dtype=np.uint8)
  for i, c in enumerate(seq):
    if (c == 'A'):
      a[i] = 0
    elif (c == 'C'):
      a[i] = 1
    elif (c == 'G'):
      a[i] = 2
    elif (c == 'T'):
      a[i] = 3
    elif (c == '-'):
      a[i] = 4
  return a

recoded_seq = []
for i in range(sequences.shape[0]):
  recoded_seq += [recode_seq(sequences[i])]

recoded_seq = np.array(recoded_seq)

def create_unaligned_sequence(sequence, left_margin, right_margin, total_columns):
    """Create the unaligned version of a sequence."""
    total_margin = left_margin + right_margin

    # Remove left margin
    del sequence[0:left_margin]

    # Adjust right margin
    sequence = sequence[:total_columns - total_margin] + ['-'] * total_margin
    return sequence


def apply_margins(sequence, left_margin, right_margin):
    """Apply left and right margins to a sequence."""
    sequence = ['-'] * left_margin + sequence[left_margin:]
    sequence = sequence[:len(sequence) - right_margin] + ['-'] * right_margin
    return sequence


def simulate_alignment_generator(rows,columns,margin_size):
  while True:
    sequence_array = make_sequences(rows,columns) #Generate the base sequences

    aligned_sequences = []
    unaligned_sequences = []

    for i in range(rows):
      aligned_sequence = list(sequence_array[i])
      unaligned_sequence = list(sequence_array[i])

      # Skip random rows
      if i in random.sample(range(rows), k=skip_rows):
        aligned_sequences.append(''.join(aligned_sequence))
        unaligned_sequences.append(''.join(unaligned_sequence))
        continue

      # Apply margins
      left_margin = random.randint(0, margin_size)
      right_margin = random.randint(0, margin_size)
      aligned_sequence = apply_margins(aligned_sequence, left_margin, right_margin)

      # Create unaligned sequence
      unaligned_sequence = create_unaligned_sequence(
        unaligned_sequence, left_margin,
        right_margin, columns
      )

      # Save results
      aligned_sequences.append(''.join(aligned_sequence))
      unaligned_sequences.append(''.join(unaligned_sequence))

    # Convert lists to numpy arrays for output
    aligned_sequences = np.array(aligned_sequences)
    unaligned_sequences = np.array(unaligned_sequences)

    # Calculate min and max positions
    yield aligned_sequences, unaligned_sequences


gap_sequence = simulate_alignment_generator(rows,columns,margin)

def add_unit_gaps(sequence_array): ##Now for the unit vector.
        sequences = [] #Build the array new.
        for i in range(sequence_array.shape[0]): #Loops through all the rows of the array.
            sequence = list(sequence_array[i]) #Access the rows individually.
            seq = recode_seq(sequence)
            seq = tf.keras.utils.to_categorical(seq, num_classes=5) #, dtype='uint8') #The extra class is necessary; we have four nucleotides + a gap.
            sequences += [seq]
        sequences = np.array(sequences)
        return sequences

gap_sequences, shift_gap_sequences = next(gap_sequence)

unit_gap_seq = add_unit_gaps(gap_sequences)
unit_shift_gap_seq = add_unit_gaps(shift_gap_sequences)


train_no_shift = np.empty((alignments,rows,columns,5),dtype='uint8') #For the unshifted gaps.
train_shift = np.empty((alignments,rows,columns,5),dtype='uint8') #For the shifted gaps.

for i in range(alignments):
  gap_sequences, shift_gap_sequences = next(gap_sequence)
  unit_gap_seq = add_unit_gaps(gap_sequences)
  unit_shift_gap_seq = add_unit_gaps(shift_gap_sequences)
  train_no_shift[i] = unit_gap_seq
  train_shift[i] = unit_shift_gap_seq

np.savez_compressed(filename,x=train_shift, y=train_no_shift)
