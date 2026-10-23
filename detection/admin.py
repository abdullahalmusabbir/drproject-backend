from django.contrib import admin
from .models import DRPredictionRecord


@admin.register(DRPredictionRecord)
class DRPredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'predicted_label',
        'probability_referable',
        'entropy',
        'is_high_uncertainty',
        'created_at',
    )
    list_filter = ('predicted_label', 'is_high_uncertainty')
    readonly_fields = (
        'id', 'image', 'predicted_class', 'predicted_label',
        'probability_referable', 'entropy', 'is_high_uncertainty',
        'ip_address', 'created_at',
    )

    def has_add_permission(self, request):
        return False