from django.contrib import admin
from analytics.models import PredictionJob,MarketSignal

admin.site.register(PredictionJob)
admin.site.register(MarketSignal)