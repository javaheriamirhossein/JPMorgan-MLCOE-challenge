# your_package/__init__.py

# You can import key functions/classes here for easier access
from DeepHalo.DeepHalo_choice_learn import DeepHaloChoiceModel
from DeepHalo.Featureless_DeepHalo import DeepHaloFeatureless2D, DeepHaloFeatureless3D
from DeepHalo.Featured_DeepHalo import DeepHaloFeatured

__version__ = "1.0.0"
__all__ = ["DeepHaloChoiceModel", "DeepHaloFeatured", "DeepHaloFeatureless2D", "DeepHaloFeatureless3D"]

# Package initialization print
print(f"Initializing DeepHaloChoiceModel package v{__version__}")