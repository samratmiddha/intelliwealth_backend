from datetime import datetime ,timedelta
from analytics.models import PredictionJob
from intelliwealth_backend.tasks import predict_portfolio_task
from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response
import yfinance as yf

from .models import Portfolio, PortfolioAsset, PortfolioMetrics
from .serializers import PortfolioSerializer, PortfolioAssetSerializer, PortfolioMetricsSerializer
from .utils import predict_prices
from users.models import ClientProfile

from assets.models import Asset

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

        portfolio = Portfolio.objects.filter(user=user).first()
        initial_equity = float(data.get('initial_equity', 100000))
        horizon = int(data.get('horizon', 6))
        selected_stocks = data.get('selected_stocks')
        # risk_level=data.get('risk_level', 'medium')
        # target_return=data.get('target_return', 0)
        # notes=data.get('notes', ''),
        # #client_assets=json.loads(user.client_profile.preferred_assets),
        # name=user.username
        # client_assets=None
        # user_id=user.id
        # try:
        #     client_profile=ClientProfile.objects.get(user=user)
        #     client_assets=json.loads(client_profile.preferred_assets)
        # except:
        #     pass

        # job = PredictionJob.objects.create(
        #     portfolio=portfolio,
        #     status='pending',
        # )

        # predict_portfolio_task.delay(
        #     user_id,
        #     initial_equity,
        #     horizon,
        #     selected_stocks,
        #     job.id,
        #     risk_level,
        #     target_return,
        #     notes,
        #     name,
        #     client_assets
            
        # )

        if not selected_stocks and portfolio:
            # If no selected_stocks are provided, use the user's portfolio assets
            selected_stocks = [pa.asset.symbol for pa in portfolio.assets.all()]

        if not selected_stocks and not portfolio:
            selected_stocks = json.loads(user.client_profile.preferred_assets)

        # if not selected_stocks or not isinstance(selected_stocks, list) or not selected_stocks:
        #     return Response({'error': 'Please provide a list of stock selected_stocks.'}, status=400)

       
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
            current_equity = 0
            price_cache = {}

            for pa in portfolio.assets.all(): 
                symbol = pa.asset.symbol 

                try:
                    if symbol not in price_cache:
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period="1d")
                        price = float(hist['Close'][-1])
                        price_cache[symbol] = price
                    else:
                        price = price_cache[symbol]
                except Exception as e:
                    print("Exception in yfinance for:", symbol, "| Error:", e)
                    price = float(pa.avg_buy_price)

                current_equity += pa.quantity * price

            if current_equity == 0:
                current_equity = initial_equity

        print("yhuuu")
        # 2. Predict portfolio
        try:
            prediction = predict_prices(horizon, current_equity, selected_stocks)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

        # 3. Delete previous PortfolioAsset entries
        PortfolioAsset.objects.filter(portfolio=portfolio).delete()

        # 4. Determine buy date (first timestamp in prediction)
        try:
            buy_date = (datetime.now() + timedelta(days=1)).date()
        except Exception:
            buy_date = None

        # 5. Create new PortfolioAsset entries with allocation, quantity, avg_buy_price
        final_weights = prediction.get('weights', {})
        for ticker, percent in final_weights.items():
            try:
                asset = Asset.objects.get(symbol=ticker)
            except Asset.DoesNotExist:
                continue

            allocation_percent = percent
            # Get buy price from PCA_Actual_Prices at buy_date
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                buy_price = float(hist['Close'][-1])
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
            roi=prediction['predicted_return'],
            sharpe_ratio=prediction['predicted_sharpe_ratio'],
            annulaized_volatility=prediction['predicted_volatility'],
            final_equity=prediction.get('expected_equity', 0),
        )
        
        return Response({
            'portfolio': portfolio.id,
            'prediction': prediction
        }, status=200)

        # return Response({"message": "Prediction started", "job_id": job.id})
    
