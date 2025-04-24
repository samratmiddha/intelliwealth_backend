
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
        (12, 'Less than 1 year'),
        (36, '1 to 3 years'),
        (60, '3 to 5 years'),
        (120, '5 to 10 years'),
        (240, 'More than 10 years'),
    ]

    PERCANTAGE_SAVINGS_CHOICES = [
        (3, '0-5% of Income'),
        (8, '5-10% of Income'),
        (13, '10-15% of Income'),
        (17, '15-20% of Income'),
        (50, '20%-100% of Income'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    name = models.CharField(max_length=100, blank=True, null=True)
    risk_tolerance = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ],blank=True,null=True)
    investment_horizon= models.IntegerField(choices=INVESTMENT_HORIZON_CHOICES,blank=True,null=True)
    primary_investment_goal=models.CharField(max_length=100,choices=INVESTMENT_GOALS_CHOICES,blank=True,null=True)
    familiarity_with_market=models.IntegerField(choices=MARKET_FAMILIARITY_CHOICES,blank=True,null=True)
    percentage_comforatable_savings=models.IntegerField(choices=PERCANTAGE_SAVINGS_CHOICES,blank=True,null=True)
    age=models.IntegerField(blank=True,null=True)
    annual_income=models.IntegerField(blank=True,null=True)
    target_return = models.FloatField(blank=True,null=True)   # in percentage
    preferred_assets = models.JSONField(default=list,blank=True)  # ['stocks', 'bonds']
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    birth_date = models.DateField(blank=True, null=True)
    avatar =models.IntegerField(default=0,blank=True)

   