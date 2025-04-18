from django.db import models
from portfolios.models import Portfolio
from assets.models import Asset

class PredictionJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='prediction_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    result = models.JSONField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

class MarketSignal(models.Model):
    SIGNAL_TYPES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
        ('hold', 'Hold'),
    ]

    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    signal_type = models.CharField(max_length=10, choices=SIGNAL_TYPES)
    strength = models.FloatField()
    prediction_date = models.DateField()
