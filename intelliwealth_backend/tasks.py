import traceback
from urllib import request
from celery import shared_task
from analytics.models import PredictionJob
import time  # simulate processing
import json
from django.utils import timezone
from datetime import datetime ,timedelta
from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response
import yfinance as yf

from portfolios.models import Portfolio, PortfolioAsset, PortfolioMetrics
from portfolios.serializers import PortfolioSerializer, PortfolioAssetSerializer, PortfolioMetricsSerializer
from portfolios.utils import predict_prices

from assets.models import Asset
from users.models import User

import json

@shared_task
def predict_portfolio_task(user_id,initial_equity, horizon, selected_stocks, job_id, risk_level, target_return,notes,name,client_assets):
    try:
        job = PredictionJob.objects.get(id=job_id)
        job.status = 'running'
        job.save()

        user=User.objects.get(id=user_id)
        user_portfolio = Portfolio.objects.filter(user=user).first()

        if not selected_stocks and user_portfolio:
            # If no selected_stocks are provided, use the user's portfolio assets
            selected_stocks = [pa.asset.symbol for pa in user_portfolio.assets.all()]

        if not selected_stocks and not user_portfolio:
            selected_stocks = client_assets
       
        if not user_portfolio:
            # Create a new portfolio
            user_portfolio = Portfolio.objects.create(
                user=user,
                name=f"{name}'s Portfolio",
                risk_level=risk_level,
                target_return=target_return,
                notes=notes,
                initial_equity=initial_equity
            )
            current_equity = initial_equity
        else:
            current_equity = 0
            price_cache = {}

            for pa in user_portfolio.assets.all(): 
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
        PortfolioAsset.objects.filter(portfolio=user_portfolio).delete()

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
                portfolio=user_portfolio,
                asset=asset,
                allocation_percent=allocation_percent,
                quantity=quantity,
                avg_buy_price=buy_price
            )

        # 6. Delete previous PortfolioMetrics and create new one
        PortfolioMetrics.objects.filter(portfolio=user_portfolio).delete()
        PortfolioMetrics.objects.create(
            portfolio=user_portfolio,
            roi=prediction['predicted_return'],
            sharpe_ratio=prediction['predicted_sharpe_ratio'],
            annulaized_volatility=prediction['predicted_volatility'],
            final_equity=prediction.get('expected_equity', 0),
        )

        job.status = 'completed'
        job.result = prediction
        job.completed_at = timezone.now()
        job.save()
    
    except PredictionJob.DoesNotExist:
        # Can't find the job? Log or handle it
        pass
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()  # This prints the full traceback to the console/logs

        job.status = 'failed'
        job.result = {
            "error": str(e),
            "trace": traceback.format_exc()  # Optional: save full trace in DB
        }
        job.completed_at = timezone.now()
        job.save()
