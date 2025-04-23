from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Portfolio, PortfolioAsset, PortfolioMetrics
from .serializers import PortfolioSerializer, PortfolioAssetSerializer, PortfolioMetricsSerializer
from .utils import predict_portfolio

class PortfolioViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Only allow the authenticated user to access their own portfolio.
        return Portfolio.objects.filter(user=self.request.user)

class PortfolioAssetViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioAssetSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        portfolio = Portfolio.objects.filter(user=self.request.user).first()
        return PortfolioAsset.objects.filter(portfolio=portfolio) if portfolio else PortfolioAsset.objects.none()

class PortfolioMetricsViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioMetricsSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        portfolio = Portfolio.objects.filter(user=self.request.user).first()
        return PortfolioMetrics.objects.filter(portfolio=portfolio) if portfolio else PortfolioMetrics.objects.none()

class PortfolioPredictionView(APIView):
    """
    Runs portfolio optimization using parameters from the user profile:
      - lookback and horizon are set from ClientProfile.investment_horizon_years.
      - initial_equity is determined from the existing portfolio’s metrics (if available), otherwise a default is used.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, format=None):
        try:
            # get user profile and portfolio
            profile = request.user.profile
            portfolio = request.user.portfolio  # one-to-one relation

            # Derive lookback and horizon from the user's investment horizon.
            # For example, if investment_horizon_years is 3, you might use 3 months lookback and 3 months horizon.
            # Adjust the conversion as needed.
            lookback = 24
            horizon = profile.investment_horizon_years

            # Use previous final equity if available; otherwise default to 10000.
            if hasattr(portfolio, 'metrics') and portfolio.metrics.final_equity:
                initial_equity = portfolio.metrics.final_equity
            else:
                initial_equity = 10000.0

            result = predict_portfolio(lookback, horizon, initial_equity)

            # Optionally, update or create portfolio metrics with the new final equity.
            from .models import PortfolioMetrics
            metrics_obj, created = PortfolioMetrics.objects.get_or_create(portfolio=portfolio)
            metrics_obj.final_equity = result['final_equity']
            metrics_obj.roi = result['performance_metrics'].get("Annualized Return", 0)
            metrics_obj.sharpe_ratio = result['performance_metrics'].get("Annualized Sharpe Ratio", 0)
            metrics_obj.annulaized_volatility = result['performance_metrics'].get("Annualized Volatility", 0)
            metrics_obj.maximum_drawdown = result['performance_metrics'].get("Maximum Drawdown", 0)
            metrics_obj.save()

            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            import traceback
            return Response(
                {"error": str(e), "trace": traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )