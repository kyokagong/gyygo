from django.db import models

# Create your models here.
class ContactMe(models.Model):
    name = models.CharField(max_length=255)

    hobby = models.TextField()

    contact = models.CharField(max_length=255)