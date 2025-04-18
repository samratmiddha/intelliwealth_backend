from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import fetch_news, get_summary

class NewsView(APIView):
    def get(self, request, format=None):
        topic = request.query_params.get('topic', 'stocks')
        quantity = int(request.query_params.get('quantity', 5))
        
        # Fetch news using the provided topic
        news_list = fetch_news(topic.replace(' ', ''))
        
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