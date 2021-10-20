import numpy as np

from datasets.mnist_set import MnistSet
from models.set_vae import SetVariationalAutoEncoder
from mnist_vae import set_config
from scipy.interpolate import interp1d

if __name__ == '__main__':
    _c = set_config()
    dataset = MnistSet(_c.train_split, _c.pad_value, 20)
    vae = SetVariationalAutoEncoder(_c.encoder_latent, _c.trans_layers, _c.trans_attn_size, _c.trans_num_heads,
                                         dataset.element_size, _c.size_pred_width, _c.pad_value, dataset.max_num_elements)
    vae.compile()

    vae.built = True
    vae.load_weights('logs/checkpoints/vae/20211020-091117/85501/').expect_partial()

    num_points = np.array([200])    # lets draw with 200 points, why not
    frames_per_transition = 5

    first_embedding = None
    frames = []
    current_set = vae.sample_prior_batch(num_points)
    current_embedding = None

    for (images, sets, sizes, labels) in dataset.get_train_set().batch(1).take(20):
        new_embedding = vae.encode_set(sets, sizes).mode()

        if current_embedding is not None:
            # interpolate between the two latent spaces, drawing the generated digit at each point
            linfit = interp1d([1, frames_per_transition], np.vstack([current_embedding, new_embedding.numpy()[0]]), axis=0)
            for i in range(1, frames_per_transition):
                decoded = vae.decode_set(linfit(i), current_set, num_points).mode()
                frames.append(linfit(i))
        else:
            current_set = vae.decode_set(new_embedding, current_set, num_points).mode()
            first_embedding = new_embedding     # store the initial one so we can loop back around to the beginning

        current_embedding = new_embedding
        pass
