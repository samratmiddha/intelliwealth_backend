from django.urls import path, include
from django.contrib import admin
from rest_framework.routers import DefaultRouter
from users.views import UserViewSet, ClientProfileViewSet,GoogleLogin
from portfolios.views import PortfolioViewSet, PortfolioAssetViewSet
from news.views import NewsView
from assets.views import AssetViewSet
from analytics.views import PredictionJobViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'profiles', ClientProfileViewSet)
router.register(r'portfolios', PortfolioViewSet,basename='portfolios')
router.register(r'portfolio-assets', PortfolioAssetViewSet, basename='portfolio-assets')
router.register(r'assets', AssetViewSet)
router.register(r'predictions', PredictionJobViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),  
    path('api/', include(router.urls)),
    path('api/auth/', include('dj_rest_auth.urls')),
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),  # Signup
    path('api/auth/social/', include('allauth.socialaccount.urls')),
    path('api/auth/google/', GoogleLogin.as_view(), name='google_login'),
    path('api/news/', NewsView.as_view(), name='news'),
    

]
