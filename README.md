# self-organizing
Research into self organizing hierarchies for artificial neural networks.

## Contents

```
virtualenv -p python3 venv
source venv/bin/activate
chmod +x train.sh 
./train.sh
```

## Results
|Units|Arch|Meta-steps|Accuracy|
|----|----|----|----|
|3|Pyramidal|20|92.5|

## TODO

- [x] Refactor the training of handcrafted architectures 
- [x] Handle both single and multi-channel inputs (CIFAR10)
- [x] Add code profiler
- [ ] Refactor the meta-learner: Simulated Annealing