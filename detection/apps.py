from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        from . import model_loader
        model_loader.load_model()