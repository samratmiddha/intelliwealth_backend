from rest_framework import viewsets, permissions
from analytics.models import PredictionJob,MarketSignal
from analytics.serializers import PredictionJobSerializer,MarketSignalSerializer



class PredictionJobViewSet(viewsets.ModelViewSet):
    queryset = PredictionJob.objects.all()
    serializer_class = PredictionJobSerializer
    permission_classes = [permissions.IsAuthenticated]

