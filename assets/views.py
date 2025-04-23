from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from assets.models import Asset
from assets.serializers import AssetSerializer
from django.conf import settings
from .utils import get_stock_data, prepare_financial_data, run_langchain_query

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
        
        if asset.data:
            return Response({"result": asset.data}, status=status.HTTP_200_OK)
        else:
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
                # Persist the transformed financial data to the asset's data field
                asset.data = df.to_dict(orient="records")[0]
                asset.save()
                
                result = run_langchain_query(df, question)
                return Response({"result": result}, status=status.HTTP_200_OK)
            except Exception as e:
                import traceback
                return Response(
                    {"error": str(e), "trace": traceback.format_exc()},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )