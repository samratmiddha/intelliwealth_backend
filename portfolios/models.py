from django.db import models
from users.models import User
from assets.models import Asset

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    risk_level = models.CharField(max_length=20)
    target_return = models.FloatField()
    notes = models.TextField(blank=True)

class PortfolioAsset(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='assets')
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    allocation_percent = models.FloatField()
    quantity = models.FloatField()
    avg_buy_price = models.DecimalField(max_digits=12, decimal_places=2)

class PortfolioMetrics(models.Model):
    portfolio = models.OneToOneField(Portfolio, on_delete=models.CASCADE, related_name='metrics')
    roi = models.FloatField()
    sharpe_ratio = models.FloatField()
    annulaized_volatility = models.FloatField()
    maximum_drawdown=models.FloatField()
    final_equity = models.FloatField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
