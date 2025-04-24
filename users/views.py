from rest_framework import viewsets, permissions
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import PermissionDenied, ValidationError
from dj_rest_auth.views import LoginView
from dj_rest_auth.registration.views import SocialLoginView, RegisterView
from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from users.models import User, ClientProfile
from users.serializers import (
    RegisterSerializer, UserSerializer, ClientProfileSerializer, GoogleLoginSerializer
)
from rest_framework.decorators import action
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
import requests





class GoogleLogin(SocialLoginView):
    adapter_class = GoogleOAuth2Adapter
    serializer_class = GoogleLoginSerializer
    permission_classes = [permissions.AllowAny]  # Replace with your actual callback URL

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        user = self.user
        try:
            social_account = SocialAccount.objects.get(user=user, provider='google')
            print("Google user extra_data:", social_account.extra_data)
        except SocialAccount.DoesNotExist:
            print("No Google SocialAccount found for user.")

        if user and not user.is_google_authenticated:
            user.is_google_authenticated = True
            user.save(update_fields=["is_google_authenticated"])

        if user and not ClientProfile.objects.filter(user=user).exists():
            client_profile=ClientProfile.objects.create(user=user)
            client_profile.name = social_account.extra_data.get("name", "")
            client_profile.profile_picture = social_account.extra_data.get("picture", "")
            client_profile.save()
            print("Created ClientProfile for user:", user.email)

        return Response(UserSerializer(user).data)


class CustomLoginView(LoginView):
    def get_response(self):
        return Response(UserSerializer(self.user).data)


class CustomRegisterView(RegisterView):
    serializer_class = RegisterSerializer

    def get_response_data(self, user):
        return UserSerializer(user).data


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


    @action(detail=False, methods=['get'], url_path='whoami')
    def whoami(self, request):
        user = request.user
        if not user.is_authenticated:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)
        
        user_data = UserSerializer(user).data
    
    # Check if user has a profile
        try:
            profile =ClientProfile.objects.get(user=user)
            profile = ClientProfileSerializer(profile).data
        except ClientProfile.DoesNotExist:
            profile = None
        
        # Combine user data with profile
        response_data = {
            "user": user_data,
            "profile": profile
        }
        
        return Response(response_data) 
    
    



class ClientProfileViewSet(viewsets.ModelViewSet):
    queryset = ClientProfile.objects.all()
    serializer_class = ClientProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        user = self.request.user
        if not user.is_financial_advisor:
            raise PermissionDenied("Only financial advisors can create client profiles.")
        if not serializer.instance.user:
            raise ValidationError("Client user must be created first.")
        serializer.save()
