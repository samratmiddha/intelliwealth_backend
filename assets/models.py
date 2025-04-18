from django.db import models

class Asset(models.Model):
    ASSET_TYPES = [
        ('stock', 'Stock'),
        ('bond', 'Bond'),
        ('mutual_fund', 'Mutual Fund'),
        ('crypto', 'Cryptocurrency'),
        ('commodity', 'Commodity'),
    ]

    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=20, unique=True)
    asset_type = models.CharField(max_length=20, choices=ASSET_TYPES)
    exchange = models.CharField(max_length=50)
    metadata = models.JSONField(default=dict,blank=True,null=True)
    data = models.JSONField(blank=True,null=True)

    def __str__(self):
        return f"{self.name} ({self.symbol}) {self.id}"

