from django.contrib import admin
from .models import CarInformation, Classification

# Register the Classification model to appear in the admin interface
@admin.register(Classification)
class ClassificationAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')  # Show these fields in the list view
    search_fields = ('name',)  # Allow searching by name
    list_filter = ('name',)  # Allow filtering by name

# Register the CarInformation model to appear in the admin interface
from django.utils.html import format_html

@admin.register(CarInformation)
class CarInformationAdmin(admin.ModelAdmin):
    list_display = ('plate_number', 'car_owner', 'address', 'classification', 'remarks', 'picture_link')

    def picture_link(self, obj):
        if obj.picture:  # Check if a picture is uploaded
            return format_html('<a href="{}" target="_blank">Open Picture</a>', obj.picture.url)
        return "No Picture"
    
    picture_link.short_description = "Picture"  # Column header in the admin


