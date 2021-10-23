import numpy as np

from datasets.mnist_set import MnistSet
from models.set_vae import SetVariationalAutoEncoder
from mnist_vae import set_config
from scipy.interpolate import interp1d

from visualisation.gif_maker import create_xy_gif

if __name__ == '__main__':
    _c = set_config()
    dataset = MnistSet(_c.train_split, _c.pad_value, 20)
    vae = SetVariationalAutoEncoder(_c.encoder_latent, _c.trans_layers, _c.trans_attn_size, _c.trans_num_heads,
                                         dataset.element_size, _c.size_pred_width, _c.pad_value, dataset.max_num_elements)
    vae.compile()

    vae.built = True
    vae.load_weights('logs/checkpoints/vae/20211020-091117/85501/').expect_partial()

    num_points = 300   # lets draw with 300 points, why not
    num_points_np = np.array([num_points])
    frames_per_transition = 15

    first_embedding = None
    frames = []
    current_embedding = None

    for (images, sets, sizes, labels) in dataset.get_train_set().shuffle(500).batch(1).take(20):
        new_embedding = vae.encode_set(sets, sizes).mode()

        if current_embedding is not None:
            # interpolate between the two latent spaces, drawing the generated digit at each point
            linfit = interp1d([1, frames_per_transition], np.vstack([current_embedding.numpy()[0], new_embedding.numpy()[0]]), axis=0)
            for i in range(1, frames_per_transition + 1):
                interp_latent = np.expand_dims(linfit(i), 0).astype(np.float32)
                initial_set = vae.sample_prior_batch(num_points_np)
                current_set = vae.decode_set(interp_latent, initial_set, num_points_np).mode()
                frames.append(current_set[0])

            # spend a bit of time refining the digit as well
        else:
            initial_set = vae.sample_prior_batch(num_points_np)
            current_set = vae.decode_set(new_embedding, initial_set, num_points_np).mode()
            first_embedding = new_embedding     # store the initial one so we can loop back around to the beginning

        current_embedding = new_embedding

    gif = create_xy_gif(frames, num_points)
    pass
