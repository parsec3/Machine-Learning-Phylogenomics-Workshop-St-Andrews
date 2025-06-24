# Machine-Learning-Phylogenomics-Workshop-St-Andrews

Welcome to our hands-on workshop on machine learning for alignments and phylogenomics.

In this assignment, you will train a simple neural network to classify the Iris dataset and then, you will train a convolutional neural network to do multiple sequence alignment.

## Datasets

In this assignment, we will use two datasets. One is the famous Iris flower data dataset compiled by the British mathematician and polymath Ronald Fisher. It is a multivariate dataset that consists of 50 samples from three species: _Iris setosa_, _I. virginica_, and _I. versicolor_. Four features (sepal and petal length and width) are used for species discrimination.

It is contained in the sklearn library under sklearn.datasets.load_iris().

You can use some simple Pandas functions to gain information about the data:

```
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data.head()

print("Describing the data: ",data.describe())
print("Info of the data:",data.info())
```

```
print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))
```

The other dataset is one we will have to simulate ourselves, but, don't worry, it won't be difficult.

We want to train a neural network to do multiple sequence alignment, so we will simulate unaligned (our features) and aligned (our labels) DNA sequence matrices through the script AlignmentSim.py. It takes the following arguments in this order: rows columns margin_size skipped_rows number_of_alignments file_name

```
python3 AlignmentSim.py 8 32 4 3 10000 Dataset.npz
```

Then, we load the dataset in Ali-U-Net.py to train our model. We can also make a second dataset of prediction data to make predictions in our AliU_Pred.py script:

```
python3 AlignmentSim.py 8 32 4 3 1000 Pred_Dataset.npz
```

## Iris Model

For our Iris dataset, we use a simple sequence of dense layers as our model:

```
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils
```

```
model = Sequential([
    Dense(1000, input_dim=4, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

The model's labels are the species names normalized as one-hot vectors:

_I. setosa_ → 0 → [1. 0. 0.]

_I. versicolor_ → 1 → [0. 1. 0.]

_I. virginica_ → 2 → [0. 0. 1.]

Meanwhile, its features are the sepal and petal measurements scaled to a unit norm.

## Ali-U-Net

For the multiple-sequence alignment, we use a convolutional neural network in the form of a modified U-Net. Its encoder path downsamples the input matrices for feature extraction while its decoder path upsamples them again.

Its inputs and outputs are, respectively, unaligned and aligned DNA nucleotide matrices one-hot encoded according to the following code:

```
[ 
 [1, 0, 0, 0, 0], # A 
 [0, 1, 0, 0, 0]  # C 
 [0, 0, 1, 0, 0], # G 
 [0, 0, 0, 1, 0], # T 
 [0, 0, 0, 0, 1], # - 
]
```

For a more detailed explanation, see this repository:

https://github.com/parsec3/Ali-U-Net
