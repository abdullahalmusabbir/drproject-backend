import os
from django.conf import settings
from .ml_engine.inference_numpy import load_model_package

_MODEL_PACKAGE = None


def load_model():
    global _MODEL_PACKAGE
    if _MODEL_PACKAGE is None:
        pkl_path = os.path.join(settings.BASE_DIR, 'detection', 'ml_engine', 'dr_model_export.pkl')
        _MODEL_PACKAGE = load_model_package(pkl_path)
    return _MODEL_PACKAGE


def get_model():
    global _MODEL_PACKAGE
    if _MODEL_PACKAGE is None:
        return load_model()
    return _MODEL_PACKAGE