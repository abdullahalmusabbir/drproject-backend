from rest_framework import serializers
from .models import DRPredictionRecord


class DRImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)


class DRPredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DRPredictionRecord
        fields = [
            'id',
            'predicted_class',
            'predicted_label',
            'probability_referable',
            'entropy',
            'is_high_uncertainty',
            'created_at',
        ]