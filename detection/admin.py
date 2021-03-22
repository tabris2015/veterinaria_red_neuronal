from django.contrib import admin
from django.utils.html import mark_safe
from .models import Detection

admin.site.site_header = 'Deteccion de razas'


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    readonly_fields = ['detection_image']

    def detection_image(self, obj):
        return mark_safe('<img src="{url}" width="200" height="200" />'.format(
            url=obj.picture.url
            )
        )
