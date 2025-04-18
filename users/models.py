
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    email = models.EmailField(unique=True)
    is_google_authenticated = models.BooleanField(default=False)
    is_financial_advisor = models.BooleanField(default=True)
    is_client=models.BooleanField(blank=True,default=False)
    financial_advisor=models.ForeignKey("self",blank=True,null=True,on_delete=models.SET_NULL,related_name="clients")


    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email
    
    def save(self, *args, **kwargs):
        if self.is_financial_advisor and self.is_client:
            raise ValueError("User cannot be both a financial advisor and a client.")
        super().save(*args, **kwargs)




class ClientProfile(models.Model):
    INVESTMENT_GOALS_CHOICES=[
        ('protecting_your_principal','Protecting Your Principal'),
        ('income_generation','Income Generation'),
        ('balanced_growth_and_income','Balanced Growth and Income'),
        ('Growth_oriented_investing','Growth-Oriented Investing'),
        ('aggressive_growth','Aggressive Growth'),
    ]
    MARKET_FAMILIARITY_CHOICES = [
    (0, 'Not familiar at all'),
    (1, 'Somewhat familiar'),
    (2, 'Moderately familiar'),
    (3, 'Very familiar'),
    (4, 'Expert level familiarity'),
]
    INVESTMENT_HORIZON_CHOICES = [
        (0, 'Less than 1 year'),
        (1, '1 to 3 years'),
        (2, '3 to 5 years'),
        (3, '5 to 10 years'),
        (4, 'More than 10 years'),
    ]

    PERCANTAGE_SAVINGS_CHOICES = [
        (0, '0-5% of Income'),
        (1, '5-10% of Income'),
        (2, '10-15% of Income'),
        (3, '15-20% of Income'),
        (4, '20%-100% of Income'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    risk_tolerance = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ])
    investment_horizon_years = models.IntegerField(choices=INVESTMENT_HORIZON_CHOICES)
    primary_investment_goal=models.CharField(max_length=100,choices=INVESTMENT_GOALS_CHOICES)
    familiarity_with_market=models.IntegerField(choices=MARKET_FAMILIARITY_CHOICES)
    percentage_comforatable_savings=models.IntegerField(choices=PERCANTAGE_SAVINGS_CHOICES)
    age=models.IntegerField()
    annual_income=models.IntegerField()
    target_return = models.FloatField()   # in percentage
    preferred_asset_types = models.JSONField(default=list)  # ['stocks', 'bonds']
   