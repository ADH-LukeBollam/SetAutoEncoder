import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions

class RandomPointSet:
    def __init__(self, pad_value):
        self.element_size = 2
        self.max_num_elements = 30
        self.pad_value = pad_value
        self.image_size = 28

        dist = tfd.Bernoulli(probs=tf.constant(0.01, tf.float32, (28, 28, 1)))
        self.point_dist = tfd.Independent(dist, 1)

    def get_points_ds(self):
        def gen():
            while(True):
                yield self.point_dist.sample()

        dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(self.image_size, self.image_size, 1), dtype=tf.int32)))
        return dataset

    def pixels_to_set(self, pixels):
        xy = tf.squeeze(pixels)
        pixel_indices = (tf.where(tf.greater(xy, tf.constant(0, dtype=tf.int32))) / 28)[:self.max_num_elements]     # no more than 30 values
        size = tf.shape(pixel_indices)[0]
        paddings = [[0, self.max_num_elements - tf.shape(pixel_indices)[0]], [0, 0]]
        padded = tf.cast(tf.pad(pixel_indices, paddings, 'CONSTANT', self.pad_value), tf.float32)
        return xy, padded, size, 0

    def get_train_set(self):
        ds = self.get_points_ds().take(100000)
        assert isinstance(ds, tf.data.Dataset)
        ds = ds.map(lambda row: self.pixels_to_set(row))
        ds = ds.filter(lambda xy, padded, size, label: size > 0)
        return ds

    def get_val_set(self):
        ds = self.get_points_ds().take(10000)
        assert isinstance(ds, tf.data.Dataset)
        ds = ds.map(lambda row: self.pixels_to_set(row))
        ds = ds.filter(lambda xy, padded, size, label: size > 0)
        return ds

    def get_test_set(self):
        ds = self.get_points_ds().take(10000)
        assert isinstance(ds, tf.data.Dataset)
        ds = ds.map(lambda row: self.pixels_to_set(row))
        ds = ds.filter(lambda xy, padded, size, label: size > 0)
        return ds


if __name__ == '__main__':
    train = RandomPointSet(-999).get_train_set()

    for sample in train.take(-1):
        raw = sample[0].numpy()
        pixel = sample[1].numpy()
        size = sample[2].numpy()
        x = pixel[:, 1]
        y = pixel[:, 0]
        plt.axis([0, 1, 0, 1])
        plt.imshow(raw)
        plt.scatter(x, y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
