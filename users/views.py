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


class GoogleLogin(SocialLoginView):
    adapter_class = GoogleOAuth2Adapter
    serializer_class = GoogleLoginSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        user = self.user

        if user and not user.is_google_authenticated:
            user.is_google_authenticated = True
            user.save(update_fields=["is_google_authenticated"])

        return Response(UserSerializer(user).data)


class CustomLoginView(LoginView):
    def get_response(self):
        return Response(UserSerializer(self.user).data)


class CustomRegisterView(RegisterView):
    serializer_class = RegisterSerializer

    def get_response_data(self, user):
        return UserSerializer(user).data


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


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
