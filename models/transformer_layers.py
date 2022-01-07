import tensorflow as tf


# implementation based off https://www.tensorflow.org/tutorials/text/transformer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform', use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform', use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform', use_bias=False)

        self.ff = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform', use_bias=False)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.ff(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk to stabilise gradients
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class PointwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(PointwiseFeedforward, self).__init__()
        self.ff1 = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self. ff2 = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform')

    def call(self, x):
        x = self.ff1(x)
        x = tf.nn.leaky_relu(x)
        x = self.ff2(x)
        return x


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(TransformerLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointwiseFeedforward(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, q, k, mask):
        attn_output, _ = self.mha(q, k, k, mask)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(q + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class PoolingMultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_seed_vectors, num_heads):
        super().__init__()
        self.mab = TransformerLayer(d_model, num_heads)
        self.seed_vectors = tf.Variable(tf.random.normal([1, num_seed_vectors, d_model]))

    def call(self, x, mask):
        b = tf.shape(x)[0]
        s = self.seed_vectors
        s = tf.tile(s, [b, 1, 1])  # shape [b, k, d]

        return self.mab(s, x, mask)


class PICASO(tf.keras.layers.Layer):
    def __init__(self, d_model, num_seed_vectors, num_heads):
        super(PICASO, self).__init__()
        self.mab = TransformerLayer(d_model, num_heads)      #shared parameters
        self.seed_vectors = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, num_seed_vectors, d_model)))

    def call(self, x, mask):
        b = tf.shape(x)[0]
        s = self.seed_vectors
        s = tf.tile(s, [b, 1, 1])  # shape [b, k, d]

        S_prime = self.mab(s, x, mask)
        S_prime2 = self.mab(S_prime, x, mask)
        S_prime3 = self.mab(S_prime2, x, mask)

        return S_prime3


class GeneralisedPICASO(tf.keras.layers.Layer):
    def __init__(self, d_model, num_seed_vectors, num_heads):
        super(GeneralisedPICASO, self).__init__()
        self.seed_vectors = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, num_seed_vectors, d_model)))

        self.mab = TransformerLayer(d_model, num_heads)
        self.mab0 = TransformerLayer(d_model, num_heads)
        self.mab1 = TransformerLayer(d_model, num_heads)
        self.mab2 = TransformerLayer(d_model, num_heads)
        self.mab3 = TransformerLayer(d_model, num_heads)
        self.mab4 = TransformerLayer(d_model, num_heads)
        self.mab5 = TransformerLayer(d_model, num_heads)


    def call(self, x, mask):
        b = tf.shape(x)[0]
        s = self.seed_vectors
        s = tf.tile(s, [b, 1, 1])  # shape [b, k, d]

        X_mask = tf.zeros((b, 1, 1, 1), x.dtype)

        H = self.mab(s, x, mask)
        X_prime = self.mab0(x, H, X_mask)

        H_prime = self.mab1(H, X_prime, mask)
        X_prime2 = self.mab2(X_prime, H_prime, X_mask)

        H_prime2 = self.mab3(H_prime, X_prime2, mask)
        X_prime3 = self.mab4(X_prime2, H_prime2, X_mask)

        S_prime = self.mab5(H_prime2, X_prime3, mask)
        return S_prime
