from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import numpy as np


def get_hungarian(point_set_a, point_set_b, sizes):
    a = tf.expand_dims(point_set_a, axis=-2)
    b = tf.expand_dims(point_set_b, axis=-3)

    largest_unpadded_dim = tf.reduce_max(sizes)
    row_sizes = tf.repeat(sizes, sizes)
    a = a[:, :largest_unpadded_dim, :largest_unpadded_dim, :]
    b = b[:, :largest_unpadded_dim, :largest_unpadded_dim, :]
    a = tf.RaggedTensor.from_tensor(tf.repeat(a, largest_unpadded_dim, -2), lengths=(sizes, row_sizes))
    b = tf.RaggedTensor.from_tensor(tf.repeat(b, largest_unpadded_dim, -3), lengths=(sizes, row_sizes))

    square_distances = tf.keras.losses.huber(a, b)

    def hungarian(sample):
        sample = sample.to_tensor()
        row_ind, col_ind = linear_sum_assignment(sample)
        inds = tf.stack([row_ind, col_ind], axis=1)
        return tf.reduce_sum(tf.gather_nd(sample, inds))

    losses = tf.map_fn(hungarian, square_distances, fn_output_signature=tf.float32)
    # s = tf.shape(sizes)[0]
    # losses = tf.TensorArray(tf.float32, s)
    # for i, r in enumerate(square_distances):
    #     r_t = r.to_tensor()
    #     loss = tf.py_function(hungarian, [r_t], tf.float32)
    #     losses.write(i, loss)

    return tf.reduce_mean(losses)

    # def hungarian(x):
    #     sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    #     for i in range(x.shape[0]):
    #         sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    #     return sol



    # listperms = tf.py_function(hungarian, [square_distances], tf.int32)
    return listperms


if __name__ == '__main__':
    set_a = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8]], [[2], [3], [4], [5], [6], [7], [8], [9]]])
    set_b = tf.constant([[[8], [7], [6], [5], [4], [3], [2], [1]], [[8], [7], [6], [5], [4], [3], [2], [1]]])
    l = get_hungarian(set_a, set_b, [5, 6])
    print(l)