# IALS++ Python Binding

This repository contains a Python binding for an adaptation of the code from the [Google Research IALS repository](https://github.com/google-research/google-research/tree/master/ials). The code implements the model described in the paper [iALS++: Speeding up Matrix Factorization with Subspace Optimization](https://arxiv.org/abs/2110.14044).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

IALS++ is an efficient matrix factorization algorithm for collaborative filtering, designed to speed up the training process using subspace optimization techniques. This project provides a Python binding to the C++ implementation of IALS++, making it easier to integrate into Python-based data processing and machine learning workflows.

## Features

- Efficient matrix factorization using IALS++.
- Python binding for easy integration with Python projects.
- Functions to save and load the trained model.
- Evaluation metrics for recommendation performance.

## Installation

Make sure to install the necessary dependencies:

For MacOS:
```bash
brew install eigen
brew install nlohmann-json
```

For Debian-based Linux:
```bash
sudo apt update
sudo apt install libeigen3-dev nlohmann-json3-dev
```

Finally, you can install the package directly from the GitHub repository.

```bash
pip install git+https://github.com/issilva5/ialspp_python.git
```

## Usage

### Importing the Package

```python
import ialspp
```

### Loading Data

Create a `Dataset` object from your data file. The data must have two columns: user id and item id.

```python
train_data = ialspp.Dataset('path/to/train.csv')
test_train_data = ialspp.Dataset('path/to/test_train.csv')
test_test_data = ialspp.Dataset('path/to/test_test.csv')
```

### Creating and Training the Recommender

```python
recommender = ialspp.IALSppRecommender(
    embedding_dim=16,
    num_users=train_data.max_user() + 1,
    num_items=train_data.max_item() + 1,
    regularization=0.0001,
    regularization_exp=1.0,
    unobserved_weight=0.1,
    stddev=0.1,
    block_size=128
)
p = recommender.Train(train_data)
print(recommender.ComputeLosses(train_data, p)) # Print losses information
```

### Evaluating the Recommender

```python
metrics = recommender.EvaluateDataset(test_train_data, test_test_data.by_user())
print(f"Rec20={metrics[0]:.4f}, Rec50={metrics[1]:.4f}, NDCG100={metrics[2]:.4f}")
```

### Saving and Loading the Model

```python
recommender.SaveModel('path/to/model.bin')
loaded_recommender = ialspp.LoadModel('path/to/model.bin')
```

## Examples

For more detailed examples and use cases, please refer to the `examples` directory in the repository.

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
