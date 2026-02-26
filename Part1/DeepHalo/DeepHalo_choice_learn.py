import tensorflow as tf
from choice_learn.models.simple_mnl import SimpleMNL 
from choice_learn.data import ChoiceDataset
from .Featured_DeepHalo import DeepHaloFeatured
from .Featureless_DeepHalo import DeepHaloFeatureless2D, DeepHaloFeatureless3D



# ============ DeepHalo Choice Model ================

class DeepHaloChoiceModel(SimpleMNL):
    def __init__(
        self,
        H=7,
        depth=3,
        embed=16,
        dropout=0.0,
        lr=1e-3,
        epochs=50,
        batch_size=64,
        featureless=False,
        add_exit_choice=False,
        optimizer='Adam',
        block_type='qua',
        loss_name='nll',
        feature2D=False,
        **kwargs,
    ):
        # Initialize ChoiceModel 
        super().__init__(optimizer=optimizer, epochs=epochs, lr=lr, batch_size=batch_size, add_exit_choice=add_exit_choice, **kwargs)
       
        self.H = H
        self.depth = depth
        self.embed = embed
        self.dropout = dropout
        self.featureless = featureless 
        self.block_type = block_type
        self.loss_name = loss_name
        self.lr = lr
        self.feature2D = feature2D
        self.epochs = epochs

        # Set  loss function to mse if chosen which works with 2D items features
        if loss_name == 'mse':
            self.loss = tf.keras.losses.MeanSquaredError()
            
        # If featureless DeepHalo implementation required 
        if self.featureless:   
            if self.feature2D:
                self.deep_halo_core = DeepHaloFeatureless2D(
                    H=H,
                    depth=depth,
                    J=embed,
                    block_type=block_type)
            else:
                self.deep_halo_core = DeepHaloFeatureless3D(
                    H=H,
                    depth=depth,
                    J=embed,
                    block_type=block_type)
            
        # If featured DeepHalo implementation required which works with 3D items features
        else:
            self.deep_halo_core = DeepHaloFeatured(
                H=H,
                depth=depth,
                embed=embed,
                dropout=dropout,
                block_type=block_type)


    # Choice-Learn will look at this property when computing gradients
    @property
    def trainable_weights(self):
        return self.deep_halo_core.trainable_weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices):
        
        # Check if items features and availability data are given
        assert items_features_by_choice is not None, "items_features_by_choice is required" 
        assert available_items_by_choice is not None, "available_items_by_choice is required"


        # If it is a tuple
        if isinstance(items_features_by_choice, (tuple, list)):
            X = items_features_by_choice[0]
        else:
            X = items_features_by_choice   # [B, J, D]  
            

        # If it is a tuple
        if isinstance(available_items_by_choice, (tuple, list)):
            avail = available_items_by_choice[0]
        else:
            avail = available_items_by_choice  # [B, J]
            

        # Called only inside train_step with training=True
        utilities = self.deep_halo_core(X, avail, training=True)  # utilities are the logits
        return utilities

   
   
    # Get the model configurations
    def get_config(self):
        return {
            "H": self.H,
            "depth": self.depth,
            "embed": self.embed,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer_name": self.optimizer_name,
            "block_type": self.block_type,
            "add_exit_choice": self.add_exit_choice,
            "featureless": self.featureless,
            "loss_name": self.loss_name,
            "feature2D": self.feature2D
        }

