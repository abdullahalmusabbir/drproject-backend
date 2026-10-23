import uuid
from django.db import models


class DRPredictionRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')

    predicted_class = models.IntegerField()
    predicted_label = models.CharField(max_length=50)
    probability_referable = models.FloatField()
    entropy = models.FloatField()
    is_high_uncertainty = models.BooleanField(default=False)

    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.id} - {self.predicted_label}"