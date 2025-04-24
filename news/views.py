from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import fetch_news, get_summary
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import os


class NewsView(APIView):
    def get(self, request, format=None):
        topic = request.query_params.get('topic', 'stocks')
        quantity = int(request.query_params.get('quantity', 5))
        
        # Fetch news using the provided topic
        news_list = fetch_news(topic)
        
        summaries = []
        for news in news_list[:quantity]:
            title, summary, image_url = get_summary(news.link.text)
            summaries.append({
                'title': news.title.text,
                'link': news.link.text,
                'summary': summary,
                'image': image_url,
                'source': news.source.text if news.source else '',
                'pubDate': news.pubDate.text
            })
        return Response({'summaries': summaries}, status=status.HTTP_200_OK)
    




# Load tweets.csv once at startup
TWEETS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'tweets.csv')
tweets_df = pd.read_csv(TWEETS_CSV_PATH, parse_dates=['Date'])
tweets_df['Date'] = tweets_df['Date'].dt.tz_localize(None)
analyzer = SentimentIntensityAnalyzer()

class TweetSentimentAnalysisView(APIView):
    """
    GET /api/news/sentiment/?date=YYYY-MM-DD
    POST /api/news/sentiment/ { "date": "YYYY-MM-DD" }
    """
    def get(self, request):
        date_str = request.query_params.get('date')
        return self._analyze(date_str)

    def post(self, request):
        date_str = request.data.get('date')
        return self._analyze(date_str)

    def _analyze(self, date_str):
        if not date_str:
            return Response({'error': 'Missing date parameter.'}, status=400)
        try:
            end_date = parser.parse(date_str).date()
        except Exception:
            return Response({'error': 'Invalid date. Use YYYY-MM-DD.'}, status=400)

        start_date = end_date - timedelta(days=7)
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        mask = (tweets_df['Date'] >= start_dt) & (tweets_df['Date'] <= end_dt)
        window_df = tweets_df.loc[mask].copy()

        if window_df.empty:
            return Response({'error': f'No tweets found between {start_date} and {end_date}.'}, status=404)

        window_df['sentiment'] = window_df['Tweet'].apply(
            lambda txt: analyzer.polarity_scores(str(txt))['compound']
        )

        grouped = window_df.groupby('Stock_Name', as_index=False)['sentiment'].mean()
        top5 = grouped.nlargest(5, 'sentiment')
        bottom5 = grouped.nsmallest(5, 'sentiment')

        def fetch_price(tkr):
            try:
                return yf.Ticker(tkr).info.get('regularMarketPrice')
            except Exception:
                return None

        top5['price'] = top5['Stock_Name'].apply(fetch_price)
        bottom5['price'] = bottom5['Stock_Name'].apply(fetch_price)

        results = {
            'top5': top5.to_dict(orient='records'),
            'bottom5': bottom5.to_dict(orient='records'),
            'start_date': str(start_date),
            'end_date': str(end_date)
        }
        return Response(results, status=200)