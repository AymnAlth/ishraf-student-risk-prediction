"""Custom exceptions for Ishraf ML system."""

class IshrafMLException(Exception):
    """Base exception for Ishraf ML service."""
    pass

class ModelNotFoundError(IshrafMLException):
    """Raised when the trained model artifact cannot be located."""
    pass

class InvalidFeatureError(IshrafMLException):
    """Raised when input data violates the feature contract or contains forbidden columns."""
    pass

class InferenceError(IshrafMLException):
    """Raised when inference calculation fails."""
    pass
