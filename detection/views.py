import numpy as np
import cv2

from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework import generics

from .serializers import DRImageUploadSerializer, DRPredictionResultSerializer
from .models import DRPredictionRecord
from .throttles import TenSecondThrottle
from .model_loader import get_model
from .ml_engine.inference_numpy import predict_from_package


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def is_likely_retina_fundus(image_bgr):
    """
    Heuristic check: does this image look like a retinal fundus image?
    Returns (ok: bool, reason: str)
    """
    if image_bgr is None:
        return False, "Invalid image file."

    h, w = image_bgr.shape[:2]

    # 1. Minimum resolution
    if min(h, w) < 224:
        return False, "Image resolution is too low for a fundus image."

    # 2. Aspect ratio — fundus images are usually near-square
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False, "Image aspect ratio does not look like a retinal fundus image."

    # 3. Color check — fundus images are colorful and typically reddish/orange
    b, g, r = cv2.split(image_bgr)
    mean_b = float(b.mean())
    mean_g = float(g.mean())
    mean_r = float(r.mean())

    # Reject near-grayscale images
    if abs(mean_r - mean_g) < 6 and abs(mean_g - mean_b) < 6:
        return False, "Image appears grayscale; not a color fundus image."

    # Reject if red channel is not dominant (fundus images are reddish)
    if mean_r <= mean_b:
        return False, "Image does not look like a retinal fundus image (expected reddish tone)."

    return True, ""


class PredictDRView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    throttle_classes = [TenSecondThrottle]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = DRImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data['image']

        file_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)

        # NEW: reject non-fundus images BEFORE running the model
        ok, reason = is_likely_retina_fundus(image_bgr)
        if not ok:
            image_file.seek(0)
            return Response({"error": reason}, status=status.HTTP_400_BAD_REQUEST)

        image_file.seek(0)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model_package = get_model()
        result = predict_from_package(image_rgb, model_package)

        try:
            with transaction.atomic():
                record = DRPredictionRecord.objects.create(
                    image=image_file,
                    predicted_class=result['predicted_class'],
                    predicted_label=result['predicted_label'],
                    probability_referable=result['probability_referable'],
                    entropy=result['entropy'],
                    is_high_uncertainty=result['is_high_uncertainty'],
                    ip_address=get_client_ip(request),
                )
        except Exception as e:
            return Response(
                {"error": "Could not save prediction", "detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        output = DRPredictionResultSerializer(record)
        return Response(output.data, status=status.HTTP_201_CREATED)
    
class PredictionListView(generics.ListAPIView):
    queryset = DRPredictionRecord.objects.all()
    serializer_class = DRPredictionResultSerializer
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = []  # লিস্ট ভিউতে থ্রটলিং বন্ধ রাখতে চাইলে

    def get_queryset(self):
        queryset = super().get_queryset()
        date = self.request.query_params.get('date', None)
        if date:
            queryset = queryset.filter(created_at__date=date)
        return queryset