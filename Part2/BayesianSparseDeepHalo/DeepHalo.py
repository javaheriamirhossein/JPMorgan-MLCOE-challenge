# ============================================================
# Encoder (DeepHalo) with Context Features as Output
# ============================================================
import tensorflow as tf

class NonlinearMap(tf.keras.layers.Layer):
    def __init__(self, H, embed=16, dropout=0.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(embed * H, dtype=tf.float64)
        self.fc2 = tf.keras.layers.Dense(embed * H, dtype=tf.float64)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float64)

    def call(self, X, training=False):
        X = self.fc1(X)
        X = tf.nn.relu(X)
        X = self.dropout(X, training=training)
        X = self.fc2(X)
        return self.norm(X)


class DeepHaloEncoder(tf.keras.Model):
    def __init__(self, H=7, depth=3, embed=16, dropout=0.0, block_type="qua", out_dim=2):
        super().__init__()
        self.H = H
        self.embed = embed
        self.block_type = block_type

        self.init_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embed, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, dtype=tf.float64),
        ])
        self.enc_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float64)

        self.aggregate_linear = [
            tf.keras.layers.Dense(H, dtype=tf.float64) for _ in range(depth)
        ]
        self.nonlinear = [
            NonlinearMap(H, embed, dropout) for _ in range(depth)
        ]
        self.project = tf.keras.layers.Dense(out_dim, dtype=tf.float64)

    def call(self, X, avail, training=False):
        Z = self.init_encoder(X, training=training)
        Z = self.enc_norm(Z)
        X_embd = Z
        avail_f = tf.cast(avail, tf.float64)

        for fc, phi_map in zip(self.aggregate_linear, self.nonlinear):
            if self.block_type == "lin":
                fcZ = fc(Z)
            elif self.block_type == "qua":
                fcZ = fc(tf.square(Z))
            else:
                raise ValueError("block_type must be 'lin' or 'qua'")

            fcZ_masked = fcZ * avail_f[..., tf.newaxis]
            sum_fc = tf.reduce_sum(fcZ_masked, axis=1)
            num_avail = tf.maximum(tf.reduce_sum(avail_f, axis=1), 1.0)
            Zbar = sum_fc / num_avail[:, tf.newaxis]

            Zbar_exp = Zbar[:, tf.newaxis, :]
            phi = phi_map(X_embd, training=training)
            phi_rs = tf.reshape(phi, (tf.shape(phi)[0], tf.shape(phi)[1], self.H, self.embed))
            delta = tf.reduce_mean(phi_rs * Zbar_exp[..., tf.newaxis], axis=2)

            Z = Z + delta

        return self.project(Z)