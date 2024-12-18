from django.db import models

class ReferenceImage(models.Model):
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='reference_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
