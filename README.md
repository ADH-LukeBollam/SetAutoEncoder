A Tensorflow 2.5 implementation of the papers:
[TSPN](https://arxiv.org/abs/2006.16841v2) - Adam R Kosiorek, Hyunjik Kim, Danilo J Rezende  
[Set Transformer](https://arxiv.org/abs/1810.00825) - Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh

<img src="interpolation.gif" alt="Image not found" width="400"/>

To train the Autoencoder, run `mnist_ae.py`

or for the variational model, run `mnist_vae.py`

To load weights from a saved step, use the `-s` argument. You can pass a specific saved step, or -1 to load the latest.

To train the Size Predictor MLP after training TSPN, use the `-p` flag in combination with `-s`

Requires:
* Python 3.7
* tensorflow 2.5
* tensorflow-probability
* tensorflow-datasets 3.2.1
* matplotlib 3.3.1
* ffmpeg library on path if exporting using `set_interpolation.py`