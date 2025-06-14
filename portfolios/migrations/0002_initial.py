# Generated by Django 5.2 on 2025-04-11 02:57

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('assets', '0001_initial'),
        ('portfolios', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='portfolio',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='portfolios', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='portfolioasset',
            name='asset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='assets.asset'),
        ),
        migrations.AddField(
            model_name='portfolioasset',
            name='portfolio',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='assets', to='portfolios.portfolio'),
        ),
        migrations.AddField(
            model_name='portfoliometrics',
            name='portfolio',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='metrics', to='portfolios.portfolio'),
        ),
    ]
