from django.db import models
from users.models import User
from assets.models import Asset
from portfolios.utils import PCA_Actual_Prices

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    risk_level = models.CharField(max_length=20)
    target_return = models.FloatField()
    notes = models.TextField(blank=True)
    initial_equity = models.FloatField(blank=True, null=True)
    

class PortfolioAsset(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='assets')
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    allocation_percent = models.FloatField()
    quantity = models.FloatField()
    avg_buy_price = models.DecimalField(max_digits=12, decimal_places=2)

    @property
    def returns(self):
        
        try:
            price_series = PCA_Actual_Prices[self.asset.symbol].dropna()
            if not price_series.empty:
                current_price = float(price_series.iloc[-1])
            else:
                current_price = float(self.avg_buy_price)
        except Exception:
            current_price = float(self.avg_buy_price)
        if self.avg_buy_price and float(self.avg_buy_price) > 0:
            return ((current_price - float(self.avg_buy_price)) / float(self.avg_buy_price))*100
        return None
    


class PortfolioMetrics(models.Model):
    portfolio = models.OneToOneField(Portfolio, on_delete=models.CASCADE, related_name='metrics')
    roi = models.FloatField()
    sharpe_ratio = models.FloatField()
    annulaized_volatility = models.FloatField()
    maximum_drawdown=models.FloatField()
    final_equity = models.FloatField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
