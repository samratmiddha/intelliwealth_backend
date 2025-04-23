from dj_rest_auth.registration.serializers import SocialLoginSerializer
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from users.models import User, ClientProfile


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    is_client = serializers.BooleanField(default=False)
    is_financial_advisor = serializers.BooleanField(default=False)

    class Meta:
        model = User
        fields = ('email', 'username', 'password', 'is_client', 'is_financial_advisor')

    def validate(self, attrs):
        if not attrs.get('is_client') and not attrs.get('is_financial_advisor'):
            raise serializers.ValidationError("User must be either client or financial advisor.")
        if attrs.get('is_client') and attrs.get('is_financial_advisor'):
            raise serializers.ValidationError("User can't be both client and financial advisor.")
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'username', 'is_client', 'is_financial_advisor', 'financial_advisor')




class ClientProfileSerializer(serializers.ModelSerializer):
    # user = UserSerializer(read_only=True)
    risk_tolerance_display = serializers.CharField(source='get_risk_tolerance_display', read_only=True)
    investment_horizon_display = serializers.CharField(source='get_investment_horizon_display', read_only=True)
    primary_investment_goal_display = serializers.CharField(source='get_primary_investment_goal_display', read_only=True)
    familiarity_with_market_display = serializers.CharField(source='get_familiarity_with_market_display', read_only=True)
    percentage_comforatable_savings_display = serializers.CharField(source='get_percentage_comforatable_savings_display', read_only=True)

    class Meta:
        model = ClientProfile
        fields = '__all__'
class GoogleLoginSerializer(SocialLoginSerializer):
    access_token = serializers.CharField(required=True, trim_whitespace=True)

    class Meta:
        model = User
        fields = ['access_token']
