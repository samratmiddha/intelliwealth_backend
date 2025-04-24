from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Portfolio, PortfolioAsset, PortfolioMetrics
from .serializers import PortfolioSerializer, PortfolioAssetSerializer, PortfolioMetricsSerializer
from .utils import predict_portfolio

from assets.models import Asset
from portfolios.utils import PCA_Actual_Prices

import json

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


class PredictPortfolioView(APIView):
    def post(self, request, *args, **kwargs):
        user = request.user
        data = request.data

        # 1. Try to get the user's first portfolio, or create if not found
        portfolio = Portfolio.objects.filter(user=user).first()
        initial_equity = float(data.get('initial_equity', 100000))
        lookback = int(data.get('lookback', 12))
        horizon = int(data.get('horizon', 6))
        tickers = data.get('tickers')

        if not tickers and portfolio:
            # If no tickers are provided, use the user's portfolio assets
            tickers = [pa.asset.symbol for pa in portfolio.assets.all()]

        if not tickers and not portfolio:
            tickers = json.loads(user.client_profile.preferred_assets)

      
        if not tickers or not isinstance(tickers, list) or not tickers:
            return Response({'error': 'Please provide a list of stock tickers.'}, status=400)

        if not portfolio:
            # Create a new portfolio
            portfolio = Portfolio.objects.create(
                user=user,
                name=f"{user.username}'s Portfolio",
                risk_level=data.get('risk_level', 'medium'),
                target_return=data.get('target_return', 0),
                notes=data.get('notes', ''),
                initial_equity=initial_equity
            )
            current_equity = initial_equity
        else:
            # Calculate current equity based on current portfolio assets and latest prices
            current_equity = 0
            latest_date = PCA_Actual_Prices.index.max()
            for pa in portfolio.assets.all():
                try:
                    price = float(PCA_Actual_Prices.loc[latest_date, pa.asset.symbol])
                except Exception:
                    print("exception in PCA_Actual_Prices:", pa.asset.symbol)
                    price = float(pa.avg_buy_price)
                current_equity += pa.quantity * price
            if current_equity == 0:
                current_equity = initial_equity

        # 2. Predict portfolio
        try:
            prediction = predict_portfolio(lookback, horizon, current_equity, tickers)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

        # 3. Delete previous PortfolioAsset entries
        PortfolioAsset.objects.filter(portfolio=portfolio).delete()

        # 4. Determine buy date (first timestamp in prediction)
        try:
            buy_date = prediction['timestamps'][0]
        except Exception:
            buy_date = None

        # 5. Create new PortfolioAsset entries with allocation, quantity, avg_buy_price
        final_weights = prediction.get('final_optimal_weights', {})
        for ticker, percent in final_weights.items():
            try:
                asset = Asset.objects.get(symbol=ticker)
            except Asset.DoesNotExist:
                continue

            allocation_percent = percent
            # Get buy price from PCA_Actual_Prices at buy_date
            try:
                buy_price = float(PCA_Actual_Prices.loc[buy_date, ticker])
            except Exception:
                buy_price = 0

            # Calculate quantity to buy
            amount_to_invest = (allocation_percent / 100.0) * current_equity
            quantity = amount_to_invest / buy_price if buy_price > 0 else 0

            PortfolioAsset.objects.create(
                portfolio=portfolio,
                asset=asset,
                allocation_percent=allocation_percent,
                quantity=quantity,
                avg_buy_price=buy_price
            )

        # 6. Delete previous PortfolioMetrics and create new one
        PortfolioMetrics.objects.filter(portfolio=portfolio).delete()
        PortfolioMetrics.objects.create(
            portfolio=portfolio,
            roi=prediction['performance_metrics'].get('Annualized Return', 0),
            sharpe_ratio=prediction['performance_metrics'].get('Annualized Sharpe Ratio', 0),
            annulaized_volatility=prediction['performance_metrics'].get('Annualized Volatility', 0),
            maximum_drawdown=prediction['performance_metrics'].get('Maximum Drawdown', 0),
            final_equity=prediction.get('final_equity', 0)
        )

        return Response({
            'portfolio': portfolio.id,
            'prediction': prediction
        }, status=200)