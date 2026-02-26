import tensorflow as tf

# ============ Featured DeepHalo Core ================


class NonlinearMap(tf.keras.layers.Layer):
    def __init__(self, H, embed=32, dropout=0.0):
        super().__init__()
        self.H = H
        self.embed = embed
        self.fc1 = tf.keras.layers.Dense(embed * H)   # [B,J,embed*H]
        self.fc2 = tf.keras.layers.Dense(embed * H)   # [B,J,embed*H]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, X, training=False):
        # X: [B, J, embed]
        X = self.fc1(X)                     # [B, J, embed*H]
        X = tf.nn.relu(X)
        X = self.dropout(X, training=training)
        X = self.fc2(X)                     # [B, J, embed*H]
        X = self.norm(X)
        return X                            # [B, J, embed*H]


class DeepHaloFeatured(tf.keras.Model):
    def __init__(self, H, depth, embed=32, dropout=0.0, block_type="qua"):
        super().__init__()
        self.H = H
        self.embed = embed

        self.init_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embed, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed),
        ])
        self.enc_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.aggregate_linear = [
            tf.keras.layers.Dense(H) for _ in range(depth)
        ]
        self.nonlinear = [
            NonlinearMap(H, embed, dropout) for _ in range(depth)
        ]
        
        self.block_type = block_type

        self.final_linear = tf.keras.layers.Dense(1)

    def call(self, X, avail, training=False):
        """
        X:      [B, J, D0]
        avail:  [B, J]
        return: logits [B, J]
        """
        # Embed items features
        Z = self.init_encoder(X, training=training)   # [B, J, embed]
        Z = self.enc_norm(Z)
        X_embd = Z                                     # [B, J, embed]

        avail_f = tf.cast(avail, tf.float32)           # [B, J]

        for fc, phi_map in zip(self.aggregate_linear, self.nonlinear):
            
            # apply linear or quadratic activation on the hidden state before aggregation
            if self.block_type == "lin":
                fcZ = fc(Z)      # [B, J, H]
            elif self.block_type == "qua":
                fcZ = fc(tf.square(Z))  # [B, J, H]
            else:
                raise ValueError(f"Unknown block type: {self.block_type}")
              
            # Sum context features over available items only              
            fcZ_masked = fcZ * avail_f[..., tf.newaxis]   # [B, J, H]

            sum_fc = tf.reduce_sum(fcZ_masked, axis=1)     # [B, H]
            num_avail = tf.reduce_sum(avail_f, axis=1)     # [B]
            num_avail = tf.maximum(num_avail, 1.0)
            Z_bar = sum_fc / num_avail[:, tf.newaxis]      # [B, H]

            # Broadcast to match folded last dim: [B,1,H] -> [B,J,H]
            Z_bar_exp = Z_bar[:, tf.newaxis, :]            # [B,1,H]

            # Nonlinear transform 
            phi = phi_map(X_embd, training=training)       # [B, J, embed*H]
            phi = phi * avail_f[..., tf.newaxis]           # [B, J, embed*H]

            # Reshape phi to separate H, then apply Z_bar, then fold back
            phi_reshaped = tf.reshape(phi, [-1, tf.shape(phi)[1], self.H, self.embed])  # [B,J,H,embed]
            
            phi_weighted = phi_reshaped * Z_bar_exp[..., tf.newaxis]  # [B,J,H,embed]
            delta = tf.reduce_mean(phi_weighted, axis=2)              # [B,J,embed]

            Z = Z + delta                                            # [B, J, embed]

        logits = self.final_linear(Z)[..., 0]                        # [B,J,1] -> [B, J]
        logits = tf.where(avail > 0, logits, -1e9)
        return logits
