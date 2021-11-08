import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfpl = tfp.layers
tfd = tfp.distributions
tfkl = tf.keras.layers
tfb = tfp.bijectors


class DeterministicSetPrior(tf.keras.Model):
    def __init__(self, event_size, max_size, *args, **kwargs):
        super(DeterministicSetPrior, self).__init__()
        self.event_size = event_size
        self.max_size = max_size

    def call(self, set_sizes):
        batch_size = tf.shape(set_sizes)[0]
        max_vals = tf.cast(self.max_size / set_sizes, tf.float32)
        start = tf.zeros_like(max_vals)

        feature_scales = tf.transpose(tf.linspace(start, max_vals, self.max_size), (1, 0))
        masking = tf.RaggedTensor.from_tensor(feature_scales, set_sizes)
        feature_scale_masked = tf.expand_dims(masking.to_tensor(shape=(batch_size, self.max_size)), -1)

        initial_set = tf.ones((batch_size, self.max_size, self.event_size)) * feature_scale_masked

        return initial_set


if __name__ == '__main__':
    event_size = 2
    set_sizes = tf.constant([109, 85, 73, 100, 124, 151])

    prior = DeterministicSetPrior(event_size, 200)
    sample = prior(set_sizes)[0]

    plt.scatter(sample[..., 0], sample[..., 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
