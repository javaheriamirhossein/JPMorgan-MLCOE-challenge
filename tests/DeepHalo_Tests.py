import os
import sys
import unittest
import tensorflow as tf
from typing import Dict, Tuple
import numpy as np
from choice_learn.data import ChoiceDataset
from choice_learn.models.simple_mnl import SimpleMNL 


project_root = os.path.abspath(os.path.join(os.path.dirname(__name__)))
sys.path.insert(0, project_root)


from DeepHalo.DeepHalo_choice_learn import DeepHaloChoiceModel
from DeepHalo.Featureless_DeepHalo import DeepHaloFeatureless2D, DeepHaloFeatureless3D
from DeepHalo.Featured_DeepHalo import DeepHaloFeatured




def make_toy_dataset(B=16, J=5, F_item=4, F_shared=3, seed=42):
    rng = np.random.RandomState(seed)

    X_shared = rng.randn(B, F_shared).astype(np.float32)          # [B,F_shared]
    X_items = rng.randn(B, J, F_item).astype(np.float32)          # [B,J,F_item]

    A = (rng.rand(B, J) < 0.8).astype(np.float32)                 # [B,J]
    A[(A.sum(axis=1) == 0), 0] = 1.0

    w_item = rng.randn(F_item).astype(np.float32)
    w_shared = rng.randn(F_shared).astype(np.float32)

    util_items = np.tensordot(X_items, w_item, axes=([2], [0]))   # [B,J]
    util_shared = (X_shared @ w_shared[:, None])                  # [B,1]
    util = util_items + util_shared                               # [B,J]

    util[A == 0] = -1e9
    choices = util.argmax(axis=1).astype(np.int32)

    return ChoiceDataset(
        items_features_by_choice=X_items,
        shared_features_by_choice=X_shared,
        available_items_by_choice=A,
        choices=choices,
        items_features_by_choice_names=[f"F_s{k}" for k in range(X_items.shape[-1])],
        shared_features_by_choice_names=[f"F_i{k}" for k in range(X_shared.shape[-1])]
    )


class TestDeepHaloChoiceModel(unittest.TestCase):

    def setUp(self):
        self.B, self.J = 16, 12
        self.F_item, self.F_shared = 4, 3
        self.dataset = make_toy_dataset(
            B=self.B, J=self.J, F_item=self.F_item, F_shared=self.F_shared
        )
        self.model = DeepHaloChoiceModel(
            H=7,
            depth=3,
            embed=16,
            dropout=0.0,
            lr=1e-3,
            epochs=50,
            batch_size=64,
            featureless=False,
            add_exit_choice=False,
            optimizer="Adam",
            block_type="qua",
            loss_name="nll",
            feature2D= False,
        )

    def _get_batch(self):
        idx = np.arange(self.B)
        X_items = self.dataset.items_features_by_choice[0][idx]   # [B,J,F_item]
        X_shared = self.dataset.shared_features_by_choice[0][idx] # [B,F_shared]
        A = self.dataset.available_items_by_choice[idx]           # [B,J]
        y = self.dataset.choices[idx]                             # [B]
        return X_items, X_shared, A, y

    # test inheritance
    def test_inheritance_from_simple_mnl(self):
        print("\n[TEST] Checking that DeepHaloChoiceModel inherits from SimpleMNL")
        self.assertIsInstance(self.model, SimpleMNL)


    
    # test model initialization    
    def test_model_initialization(self):
        print("\n[TEST] Verifying DeepHaloChoiceModel initializes and has core components")
        self.assertIsInstance(self.model, DeepHaloChoiceModel)
        self.assertIsNotNone(self.model.deep_halo_core)
        self.assertTrue(hasattr(self.model.deep_halo_core, "init_encoder"))
        self.assertTrue(hasattr(self.model.deep_halo_core, "aggregate_linear"))
        self.assertTrue(hasattr(self.model.deep_halo_core, "nonlinear"))

    # test model configuration
    def test_model_configuration(self):
        print("\n[TEST] Checking DeepHaloChoiceModel configuration and core hyperparameters")
        self.assertEqual(self.model.H, 7)
        self.assertEqual(self.model.depth, 3)
        self.assertEqual(self.model.embed, 16)
        self.assertEqual(self.model.dropout, 0.0)
        self.assertAlmostEqual(self.model.lr, 1e-3)
        self.assertEqual(self.model.epochs, 50)
        self.assertEqual(self.model.batch_size, 64)
        self.assertFalse(self.model.featureless)
        self.assertFalse(self.model.add_exit_choice)
        self.assertEqual(self.model.optimizer_name, "Adam")
        self.assertEqual(self.model.block_type, "qua")
        self.assertEqual(self.model.loss_name, "nll")
        self.assertFalse(self.model.feature2D)

        core = self.model.deep_halo_core
        self.assertEqual(core.H, 7)
        self.assertEqual(core.embed, 16)
        
        # check the number of layers if equal to the depth
        self.assertEqual(len(core.aggregate_linear), 3)
        self.assertEqual(len(core.nonlinear), 3)

    # test input / output shape and format
    def test_input_output_shapes(self):
        print("\n[TEST] Checking input and output shapes/dtypes of compute_batch_utility")
        X_items, X_shared, A, y = self._get_batch()
        logits = self.model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        self.assertEqual(logits.shape, (self.B, self.J))
        self.assertEqual(logits.dtype, tf.float32)


        # test core type vs featureless flag
    def test_core_type_matches_featureless_flag(self):
        print("\n[TEST] Checking core class matches featureless flag")
        model_feat = DeepHaloChoiceModel(featureless=False)
        self.assertIsInstance(model_feat.deep_halo_core, DeepHaloFeatured)
        model_fless3D = DeepHaloChoiceModel(featureless=True, embed=self.J)
        self.assertIsInstance(model_fless3D.deep_halo_core, DeepHaloFeatureless3D)
        model_fless2D = DeepHaloChoiceModel(featureless=True, feature2D=True, embed=self.J)
        self.assertIsInstance(model_fless2D.deep_halo_core, DeepHaloFeatureless2D)

        
    # test X shape depends on featureless flag
    def test_X_shape_depends_on_featureless_flag(self):
        print("\n[TEST] Checking X shape in compute_batch_utility for featureless vs featured")

        X_items, X_shared, A, y = self._get_batch()
        
        # featured: X should be 3D [B,J,D0]
        # Here we assert that the object we pass is 3D.
        self.assertEqual(len(X_items.shape), 3)
        
        model_feat = DeepHaloChoiceModel(featureless=False)                            
        _ = model_feat.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )

        # featureless2D: X_items should be 2D  [B,D0] 
        # Here we assert that the object we pass is 2D.      
        X_items_reduced = np.sum(X_items, axis=1)
        self.assertEqual(len(X_items_reduced.shape), 2)
        
        model_fless = DeepHaloChoiceModel(featureless=True, feature2D=True, embed=self.J)
        _ = model_fless.compute_batch_utility(
            items_features_by_choice=X_items_reduced,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )

    
    # test output utilities with availability mask
    def test_availability_mask_on_logits(self):
        print("\n[TEST] Verifying logits are masked (very negative) for unavailable items")
        X_items, X_shared, A, y = self._get_batch()
        logits = self.model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        mask_unavail = (A == 0)
        masked_logits = tf.boolean_mask(logits, mask_unavail)
        self.assertTrue(tf.reduce_all(masked_logits <= -1e8))
        mask_avail = (A == 1)
        avail_logits = tf.boolean_mask(logits, mask_avail)
        self.assertTrue(tf.reduce_any(avail_logits > -1e8))

    # test that training decreases NLL    
    def test_training_decreases_nll(self):
        print("\n[TEST] Checking that training reduces negative log-likelihood on toy data")
        X_items, X_shared, A, y = self._get_batch()

        def nll_from_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            y_oh = tf.one_hot(y, depth=probs.shape[1])
            return -tf.reduce_mean(
                tf.reduce_sum(y_oh * tf.math.log(probs + 1e-8), axis=-1)
            )

        logits0 = self.model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        nll0 = float(nll_from_logits(logits0))
        
        
        self.model.epochs = 100
        self.model.fit(self.dataset)

        logits1 = self.model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        nll1 = float(nll_from_logits(logits1))

        self.assertLessEqual(nll1, nll0)

    # test that training decreases mse    
    def test_training_decreases_mse(self):
        print("\n[TEST] Checking that training reduces mse on toy data")
        X_items, X_shared, A, y = self._get_batch()

        def mse_from_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            y_oh = tf.one_hot(y, depth=probs.shape[1])
            return tf.reduce_mean(
                tf.reduce_sum((y_oh - probs)**2, axis=-1), axis=-1)
            

        mse_model = DeepHaloChoiceModel(
            H=7,
            depth=3,
            embed=16,
            dropout=0.0,
            lr=1e-3,
            epochs=100,
            batch_size=self.B,
            featureless=False,
            add_exit_choice=False,
            optimizer="Adam",
            block_type="qua",
            loss_name="mse",
            feature2D=False,
        )

        assert isinstance(mse_model.loss, tf.keras.losses.MeanSquaredError)

        logits0 = mse_model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        mse0 = float(mse_from_logits(logits0))

        mse_model.fit(self.dataset)

        logits1 = mse_model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )
        mse1 = float(mse_from_logits(logits1))

        self.assertLessEqual(mse1, mse0)

    # test equivariance in featured setting
    def test_equivariance_under_item_permutation_featured(self):
        print("\n[TEST] Checking permutation equivariance for featured DeepHaloChoiceModel")
        model = DeepHaloChoiceModel(
            H=7,
            depth=3,
            embed=16,
            dropout=0.0,
            lr=1e-3,
            epochs=10,
            batch_size=self.B,
            featureless=False,
            add_exit_choice=False,
            optimizer="Adam",
            block_type="qua",
            loss_name="nll",
            feature2D=False,
        )

        X_items, X_shared, A, y = self._get_batch()   # X_items: [B,J,D0]  # X_shared: [B,F0]

        logits = model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )                                             # [B,J]

        perm = np.random.permutation(self.J)          # [J]
        inv_perm = np.argsort(perm)
        inv_perm_tf = tf.convert_to_tensor(inv_perm, dtype=tf.int32)

        X_items_perm = X_items[:, perm, :]            # [B,J,D0]
        A_perm = A[:, perm]

        logits_perm = model.compute_batch_utility(
            items_features_by_choice=X_items_perm,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A_perm,
            choices=y,
        )                                             # [B,J]

        # undo permutation using tf.gather
        logits_perm_unperm = tf.gather(logits_perm, inv_perm_tf, axis=1)

        max_diff = tf.reduce_max(tf.abs(logits - logits_perm_unperm))
        self.assertLess(float(max_diff), 1e-5)



    # test equivariance in 3D featureless setting
    def test_equivariance_under_item_permutation_featureless3D(self):
        print("\n[TEST] Checking permutation equivariance for 3D featureless DeepHaloChoiceModel")
        model = DeepHaloChoiceModel(
            H=7,
            depth=3,
            embed=self.J,
            dropout=0.0,
            lr=1e-3,
            epochs=10,
            batch_size=self.B,
            featureless=True,
            add_exit_choice=False,
            optimizer="Adam",
            block_type="qua",
            loss_name="nll",
            feature2D=False,
        )

        X_items, X_shared, A, y = self._get_batch()   # X_items: [B,J,D0]  # X_shared: [B,F0]
        X_items = np.tile(np.eye(self.J)[None, ...], (A.shape[0], 1, 1))  # replace items features with one-hot embeddings

        logits = model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )                                             # [B,J]

        perm = np.random.permutation(self.J)          # [J]
        inv_perm = np.argsort(perm)
        inv_perm_tf = tf.convert_to_tensor(inv_perm, dtype=tf.int32)

        X_items_perm = X_items[:, perm, :]            # [B,J,D0]
        A_perm = A[:, perm]

        logits_perm = model.compute_batch_utility(    # [B,J]
            items_features_by_choice=X_items_perm,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A_perm,
            choices=y,
        )                                             

        # undo permutation using tf.gather
        logits_perm_unperm = tf.gather(logits_perm, inv_perm_tf, axis=1)

        max_diff = tf.reduce_max(tf.abs(logits - logits_perm_unperm))
        self.assertLess(float(max_diff), 1e-5)
        
    # test invariance (different from equivariance) in 2D featureless setting
    def test_invariance_under_item_permutation_featureless2D(self):
        print("\n[TEST] Checking permutation invariance for 2D featureless DeepHaloChoiceModel")
        model = DeepHaloChoiceModel(
            H=7,
            depth=3,
            embed=self.J,
            dropout=0.0,
            lr=1e-3,
            epochs=10,
            batch_size=self.B,
            featureless=True,
            add_exit_choice=False,
            optimizer="Adam",
            block_type="qua",
            loss_name="nll",
            feature2D=True,
        )

        X_items, X_shared, A, y = self._get_batch()   # X_items: [B,J,D0]  # X_shared: [B,F0]
        X_items = np.tile(np.eye(self.J)[None, ...], (A.shape[0], 1, 1))  # replace items features with one-hot embeddings

        logits = model.compute_batch_utility(
            items_features_by_choice=X_items,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A,
            choices=y,
        )                                             # [B,J]

        perm = np.random.permutation(self.J)          # [J]
        inv_perm = np.argsort(perm)
        inv_perm_tf = tf.convert_to_tensor(inv_perm, dtype=tf.int32)

        X_items_perm = X_items[:, perm, :]            # [B,J,D0]
        A_perm = A[:, perm]

        logits_perm = model.compute_batch_utility(
            items_features_by_choice=X_items_perm,
            shared_features_by_choice=X_shared,
            available_items_by_choice=A_perm,
            choices=y,
        )                                            

    
        logits_diff = logits - logits_perm
        logits_diff_masked = logits_diff*A_perm*A   # only compare on common available indices in A and A_perm: if they are the same, then invariance holds
        
        max_diff = tf.reduce_max(tf.abs(logits_diff_masked))
        self.assertLess(float(max_diff), 1e-5)

if __name__ == "__main__":
    unittest.main()

