from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from assets.models import Asset
from assets.serializers import AssetSerializer
from django.conf import settings
from .utils import get_stock_data, prepare_financial_data, run_langchain_query

from io import BytesIO
from django.http import HttpResponse
from reportlab.pdfgen import canvas
import markdown2
from xhtml2pdf import pisa

API_KEY = getattr(settings, "FINANCIAL_API_KEY", "C1HRSweTniWdBuLmTTse9w8KpkoiouM5")

class AssetViewSet(viewsets.ModelViewSet):
    queryset = Asset.objects.all()
    serializer_class = AssetSerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=True, methods=['get'], url_path='get-report')
    def get_report(self, request, pk=None):
        asset = self.get_object()
        if asset.asset_type != 'stock':
            return Response({"error": "Asset type is not stock"}, status=status.HTTP_400_BAD_REQUEST)
        
        ticker = asset.symbol
        question = f"{ticker} Financial Report"
        
        try:
            data = get_stock_data(ticker, API_KEY)
            if not data or not any(data.values()):
                return Response(
                    {"error": f"No data found for ticker {ticker}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            df = prepare_financial_data(data)
            
            result = run_langchain_query(df, question)
            markdown_report = result["raw"]
            asset.data = asset.data or {}
            asset.data["markdown_report"] = markdown_report
            asset.save()
            return Response({"result": markdown_report}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    @action(detail=True, methods=['get'], url_path='download-report')
    def download_report(self, request, pk=None):

        asset = self.get_object()
        if asset.asset_type != 'stock':
            return Response({"error": "Asset type is not stock"}, status=status.HTTP_400_BAD_REQUEST)

        markdown_report = None
        if asset.data and asset.data.get("markdown_report"):
            markdown_report = asset.data["markdown_report"]
        else:
            return Response({"error": "No markdown report available for this asset"}, status=status.HTTP_404_NOT_FOUND)

        # Convert markdown to HTML
        html = markdown2.markdown(markdown_report)

        # Convert HTML to PDF
        buffer = BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=buffer)
        if pisa_status.err:
            return Response({"error": "Failed to generate PDF"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        buffer.seek(0)

        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{asset.symbol}_report.pdf"'
        return response