# ============================================================
# Encoder (DeepHalo) with Context Features as Output
# ============================================================
import tensorflow as tf
from typing import List, Optional


class NonlinearMap(tf.keras.layers.Layer):
    """
    A two-layer MLP with ReLU activation, dropout, and layer normalization.
    Used as the per-element nonlinear transformation phi in each encoder block.
    """

    def __init__(self, H: int, embed: int = 16, dropout: float = 0.0) -> None:
        """
        Args:
            H:       Number of aggregation heads / context features.
            embed:   Embedding dimension per head.
            dropout: Dropout rate applied between the two dense layers.
        """
        super().__init__()
        # First linear projection: maps input to (embed * H)-dimensional space
        self.fc1 = tf.keras.layers.Dense(embed * H, dtype=tf.float64)
        # Second linear projection: same output size as fc1
        self.fc2 = tf.keras.layers.Dense(embed * H, dtype=tf.float64)
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Layer norm for stable training
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float64)

    def call(self, X: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the nonlinear map.

        Args:
            X:        Input tensor of shape (batch, N, input_dim).
            training: Whether the model is in training mode (affects dropout).

        Returns:
            Normalized output tensor of shape (batch, N, embed * H).
        """
        X = self.fc1(X)               # Linear projection
        X = tf.nn.relu(X)             # Nonlinear activation
        X = self.dropout(X, training=training)
        X = self.fc2(X)               # Second linear projection
        return self.norm(X)           # Layer normalization


class DeepHaloEncoder(tf.keras.Model):
    """
    DeepHalo encoder that produces per-element context-aware embeddings.

    Each encoding block aggregates information from all available neighbors
    into a context vector (Zbar), which is then used to update each element's
    representation via a nonlinear interaction (delta).
    """

    def __init__(
        self,
        H: int = 7,
        depth: int = 3,
        embed: int = 16,
        dropout: float = 0.0,
        block_type: str = "qua",
        out_dim: int = 2,
    ) -> None:
        """
        Args:
            H:          Number of aggregation heads (context features).
            depth:      Number of encoder blocks stacked sequentially.
            embed:      Embedding dimension per head.
            dropout:    Dropout rate used in the initial encoder and NonlinearMap.
            block_type: Aggregation mode — 'lin' for linear, 'qua' for quadratic (element-wise square).
            out_dim:    Dimensionality of the final output per element.
        """
        super().__init__()
        self.H = H
        self.embed = embed
        self.block_type = block_type

        # Initial per-element MLP encoder: maps raw features to embed-dimensional space
        self.init_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embed, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, dtype=tf.float64),
        ])
        # Layer norm after initial encoding
        self.enc_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float64)

        # One linear aggregation layer per block: projects Z (or Z^2) to H context dimensions
        self.aggregate_linear: List[tf.keras.layers.Dense] = [
            tf.keras.layers.Dense(H, dtype=tf.float64) for _ in range(depth)
        ]

        # One NonlinearMap per block: computes per-element interaction with the context
        self.nonlinear: List[NonlinearMap] = [
            NonlinearMap(H, embed, dropout) for _ in range(depth)
        ]

        # Final linear projection to output space
        self.project = tf.keras.layers.Dense(out_dim, dtype=tf.float64)

    def call(
        self,
        X: tf.Tensor,       # Shape: (batch, N, input_features)
        avail: tf.Tensor,   # Shape: (batch, N) — binary mask of available elements
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass of the DeepHalo encoder.

        Args:
            X:        Input feature tensor, shape (batch, N, input_features).
            avail:    Binary availability mask, shape (batch, N).
                      1 indicates the element is present/valid, 0 means masked out.
            training: Whether in training mode.

        Returns:
            Per-element output tensor of shape (batch, N, out_dim).
        """
        # Embed each element independently with the initial MLP
        Z = self.init_encoder(X, training=training)
        Z = self.enc_norm(Z)
        X_embd = Z  # Store initial embedding for reuse as phi input

        # Cast availability mask to float64 for masked aggregation
        avail_f = tf.cast(avail, tf.float64)

        # Iteratively refine Z through depth encoder blocks
        for fc, phi_map in zip(self.aggregate_linear, self.nonlinear):
            # Compute aggregation features: linear or quadratic projection of Z
            if self.block_type == "lin":
                fcZ = fc(Z)              # Linear: fc(Z), shape (batch, N, H)
            elif self.block_type == "qua":
                fcZ = fc(tf.square(Z))   # Quadratic: fc(Z^2), shape (batch, N, H)
            else:
                raise ValueError("block_type must be 'lin' or 'qua'")

            # Mask out unavailable elements before aggregation
            fcZ_masked = fcZ * avail_f[..., tf.newaxis]  # (batch, N, H)

            # Compute masked mean over the N dimension to get context vector Zbar
            sum_fc = tf.reduce_sum(fcZ_masked, axis=1)                          # (batch, H)
            num_avail = tf.maximum(tf.reduce_sum(avail_f, axis=1), 1.0)         # (batch,) — avoid division by zero
            Zbar = sum_fc / num_avail[:, tf.newaxis]                             # (batch, H)

            # Expand Zbar for broadcasting across N elements
            Zbar_exp = Zbar[:, tf.newaxis, :]  # (batch, 1, H)

            # Apply per-element nonlinear map to the initial embedding
            phi = phi_map(X_embd, training=training)  # (batch, N, H * embed)

            # Reshape phi to separate H heads and embed dimensions
            phi_rs = tf.reshape(phi, (tf.shape(phi)[0], tf.shape(phi)[1], self.H, self.embed))
            # (batch, N, H, embed)

            # Compute context-modulated update: weight phi by Zbar, then average over H heads
            delta = tf.reduce_mean(phi_rs * Zbar_exp[..., tf.newaxis], axis=2)  # (batch, N, embed)

            # Residual update: add context-aware delta to current representation
            Z = Z + delta

        # Project final representation to output dimension
        return self.project(Z)  # (batch, N, out_dim)
