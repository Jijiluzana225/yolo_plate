from django.db import models

# Classification model to classify the car (e.g., sedan, truck, etc.)
class Classification(models.Model):
    name = models.CharField(max_length=100, unique=True)  # Name of the classification (e.g., sedan, truck)
    description = models.TextField(blank=True, null=True)  # Optional description for the classification

    def __str__(self):
        return self.name

# CarInformation model with fields Plate Number, Car Owner, Address, Classification, and Remarks
class CarInformation(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    car_owner = models.CharField(max_length=100)
    address = models.TextField()
    classification = models.ForeignKey(Classification, on_delete=models.CASCADE)
    remarks = models.TextField(blank=True, null=True)
    picture = models.ImageField(upload_to='car_pictures/', blank=True, null=True)

    def __str__(self):
        return self.plate_number