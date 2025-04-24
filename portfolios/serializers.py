from rest_framework import serializers
from .models import Portfolio, PortfolioAsset, PortfolioMetrics
from assets.serializers import AssetSerializer

class PortfolioAssetSerializer(serializers.ModelSerializer):
    asset=AssetSerializer(read_only=True)
    returns = serializers.SerializerMethodField()
    class Meta:
        model = PortfolioAsset
        fields = '__all__'

    def get_returns(self, obj):
        return obj.returns

class PortfolioMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioMetrics
        fields = '__all__'

class PortfolioSerializer(serializers.ModelSerializer):
    assets = PortfolioAssetSerializer(many=True, read_only=True)
    metrics = PortfolioMetricsSerializer(read_only=True)

    class Meta:
        model = Portfolio
        fields = '__all__'
