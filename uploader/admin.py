from django.contrib import admin
from .models import Video

@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'subcategory', 'keywords_display', 'uploaded_at']
    list_filter = ['category', 'subcategory', 'uploaded_at']
    search_fields = ['title', 'extracted_text', 'corrected_text', 'category', 'subcategory']
    readonly_fields = ['extracted_text', 'corrected_text', 'keywords', 'category', 'subcategory', 'analysis_metadata', 'uploaded_at']
    
    fieldsets = (
        ('Informations de base', {
            'fields': ('title', 'file', 'uploaded_at')
        }),
        ('Analyse OCR', {
            'fields': ('extracted_text', 'corrected_text'),
            'classes': ('collapse',),
        }),
        ('Analyse IA', {
            'fields': ('category', 'subcategory', 'keywords'),
            'classes': ('collapse',),
        }),
        ('Métadonnées', {
            'fields': ('analysis_metadata',),
            'classes': ('collapse',),
        }),
    )
    
    def keywords_display(self, obj):
        """Affiche les mots-clés dans la liste."""
        if obj.keywords and isinstance(obj.keywords, list):
            return ', '.join(obj.keywords[:3]) + ('...' if len(obj.keywords) > 3 else '')
        return '-'
    keywords_display.short_description = 'Mots-clés'
    
    def get_queryset(self, request):
        """Optimise les requêtes pour la liste."""
        return super().get_queryset(request).select_related()
