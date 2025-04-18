from django.contrib import admin
from portfolios.models import Portfolio,PortfolioAsset,PortfolioMetrics


admin.site.register(Portfolio)
admin.site.register(PortfolioAsset)
admin.site.register(PortfolioMetrics)
