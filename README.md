# self-organizing
Research into self organizing hierarchies for artificial neural networks.

## Contents

1. `handcrafted-topologies.ipynb` contains the implementation of the handcrafted architectures, logistic regression and autoencoder models. Also includes the training pipeline for the meta-network along with its sub-units.

- 3 AE units
![alt text](/images/3-units.png)

- 5 AE units
![alt text](/images/5-units.png)

- 7 AE units
![alt text](/images/7-units.png)

- Forward pass of the meta-network
![alt text](/images/feature-map.png)

2. `correlation-analysis.ipynb` contains accuracy vs entropy correlation analysis with data generated from `handcrafted-topologies.ipynb`

> Note : `correlation-analysis.ipynb` has yet to be commited.

![alt text](/images/regressed.png)


3. `meta-learning.ipynb` contains the implementation of meta-heuristics and the overall meta-learning pipeline.

- Accuracy vs Meta-step
![alt text](/images/meta-learned-1.png)

- Entropy vs Meta-step
![alt text](/images/meta-learned-2.png)
