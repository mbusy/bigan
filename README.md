# BiGAN
Bidirectional Generative Adversarial Network implementation for Pytorch. This implementation uses the MNIST dataset.

## Models
The current implentation proposes a "meta-class" BiGAN, containing three networks (`Generator`, `Encoder` and `Discriminator`). The `NetManager` class can be used to train the BiGAN (its networks), and to produce logs (tensorboard) / plot results. T-SNE can be used to visualize large latent spaces in 2D.

## Usage
Using `main.py`:
```shell
# default launch
python3 main.py

# training
python3 main.py --batch-size 64 --epochs 40 --seed 1 --log-interval 10

# loading existing weights
python3 main.py --weights weights
```

Note: you should create the following folders at the root of the repo: `logs` and `results`.
