from rest_framework import serializers
from .models import PredictionJob, MarketSignal

class PredictionJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionJob
        fields = '__all__'

class MarketSignalSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketSignal
        fields = '__all__'
