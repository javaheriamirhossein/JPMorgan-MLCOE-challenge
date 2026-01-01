import tensorflow as tf

# ============ Featureless DeepHalo Implementation ================

class LinResBlock(tf.keras.layers.Layer):
    """
    LinResBlock:
    - Input:  Z [B, H] (current hidden state for the whole set)
              X0 [B, D0]  (aggregated items feature vector for the whole set)
    - Idea:   Use a function of the whole choice set (via X0) to
              gate the hidden state Z, then apply a linear transform
              + residual.
    """
    def __init__(self, hidden_dim):
        """
        input_dim : D0  (length of aggregated items feature vector for the whole set)
        hidden_dim: H   (hidden variable length/ number of attention heads) (we assume they are the same)
        """
        super().__init__()
        
        # Linear on hidden state Z (main path)
        self.linear = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        
        # Linear on original input X0 to produce a modulation vector
        self.linear_input  = tf.keras.layers.Dense(hidden_dim, use_bias=False)

    def call(self, Z, X0):
        """
        - Input:
            Z:  [B, H]  current hidden state
            X0: [B, D0]  original availability vector
        - Output: Z' [B, H] 
        """
        Coeffs = self.linear_input(X0)   # [B, H], context from whole choice set
        Z_modulated = Z * Coeffs         # elementwise modulation: [B, H]
        Z_out = self.linear(Z_modulated) + Z  # residual update
        return Z_out


class QuaResBlock(tf.keras.layers.Layer):
    """
    QuaResBlock:
    - Input:  Z [B, H]
    - Output: Z' [B, H]  
     Apply a quadratic (elementwise square) transform followed by
     a linear layer, then add a residual connection.
    """
    def __init__(self, H):
        """
        dim: H  (hidden width)
        """
        super().__init__()
        self.linear = tf.keras.layers.Dense(H, use_bias=False)

    def call(self, Z):
        """
        Z: [B, H]
        """
        Z_sq = tf.square(Z)          # elementwise square: [B, H]
        Z_out = self.linear(Z_sq) + Z
        return Z_out


# ---------- Main featureless DeepHalo network ----------

class DeepHaloFeatureless2D(tf.keras.Model):
    """
    Featureless DeepHalo implementation with 2D items features

    - Input:  X:       2D items features [B,D0]  In practice D0=J. However, the code works for any D0 
              avail:   availability mask [B,J]
    - Output: logits over the J items, with
              entries forced to -∞ (≈ -1e9) for unavailable items.
    """
    def __init__(self, H, depth, J, block_type="qua"):
        """
        H          : hidden dimension / number of attention heads (we assume they are the same)
        depth      : total number of layers = 1 input layer + (depth-1) blocks
        block_type:  Element in {"lin","qua"},
                     specifying the type of residual blocks.
        """
        super().__init__()
        
        # Get the number of items
        self.J = J

        # Map X [B×D0] to hidden state Z [B×H] 
        self.in_lin  = tf.keras.layers.Dense(H, use_bias=False)
        
        
        # Map hidden state z (B×H) back to logits over items (B×J)
        self.out_lin = tf.keras.layers.Dense(J, use_bias=False)

        # Build the residual blocks
        self.blocks = []
        for _ in range(depth-1):
            if block_type == "lin":
                # LinResBlock sees the whole choice set via X0
                self.blocks.append(LinResBlock(H))
            elif block_type == "qua":
                # QuaResBlock  quadratic in Z
                self.blocks.append(QuaResBlock(H))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

    def call(self, X, avail, training=False):
        """
        X        : [B, D0] float32 tensor of 2D items features.
        avail    : [B, J] float32 tensor of 0/1 availability indicators.
        """
        
        # Boolean mask of available items 
        mask =  avail>0           # [B, J] bool

        # if X is [B,J,D0]
        if len(X.shape)>2:    
            # Sum over the second axis (items) to make it [B,D0]
            X = tf.reduce_sum(X * tf.expand_dims(avail, axis=-1), axis = 1)   
                 

        # Save a copy of original input for LinResBlocks (global context)
        X0 = tf.identity(X)       # [B, D0]

        # Initial hidden state from availability vector
        Z = self.in_lin(X)        # [B, H] (we assume the hidden dimension d=H)

        # Pass through the residual blocks
        for block in self.blocks:
            if isinstance(block, LinResBlock):
                Z = block(Z, X0)  # uses whole-set info from X0
            else:
                Z = block(Z)      # quadratic residual only

        # Map hidden state back to items logits
        logits_unmasked = self.out_lin(Z)  # [B, J]

        # Mask out unavailable items by setting logits to a large negative number
        logits = tf.where(mask, logits_unmasked, -1e9)

        return logits
        
        
        
class DeepHaloFeatureless3D(tf.keras.Model):
    """
    Featureless DeepHalo implementation with 3D items features

    - Input:  X:       3D items features [B,J,J]
              avail:   availability mask [B,J]
    - Output: logits over the J items, with
              entries forced to -∞ (≈ -1e9) for unavailable items.
    """
    def __init__(self, H, depth, J, block_type="qua"):
        super().__init__()
        self.H = H
        self.J = J


        self.aggregate_linear = [
            tf.keras.layers.Dense(H) for _ in range(depth)
        ]
        self.linear = [
            tf.keras.layers.Dense(J * H) for _ in range(depth)
        ]

        self.final_linear = tf.keras.layers.Dense(1)
        
        self.block_type = block_type

    def call(self, X, avail, training=False):
        """
        X:      [B, J, J]   In featureless setting we deal with one-hot vectors.
        avail:  [B, J]
        return: logits [B, J] 
        """
        assert X.shape[-1] == self.J # the embedding size should be equal to J
        


        # Here we do not use any nonlinear mapping. We use identity map 
        X0 = tf.identity(X) 
        Z = X0                                     # [B, J, J] (the hidden_dim is J here)

        # cast the availability mask into tensorflow tensor
        avail_f = tf.cast(avail, tf.float32)         # [B, J]


        for fc, phi_map in zip(self.aggregate_linear, self.linear):
        
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

            # Linear transform 
            phi = phi_map(X0, training=training)       # [B, J, J*H]
            phi = phi * avail_f[..., tf.newaxis]           # [B, J, J*H]

            # Reshape phi to separate H, then apply Z_bar, then fold back
            phi_reshaped = tf.reshape(phi, [-1, tf.shape(phi)[1], self.H, self.J])  # [B,J,H,J]
            
            phi_weighted = phi_reshaped * Z_bar_exp[..., tf.newaxis]  # [B,J,H,J]
            delta = tf.reduce_mean(phi_weighted, axis=2)              # [B,J,J]

            Z = Z + delta                                            # [B, J, J]

        logits = self.final_linear(Z)[..., 0]                        # [B,J,1] -> [B, J]
        logits = tf.where(avail > 0, logits, -1e9)
        return logits