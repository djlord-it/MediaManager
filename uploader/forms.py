from django import forms
from .models import Video

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['title', 'file']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Entrez le titre de votre vidéo...',
                'maxlength': 100
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*'
            })
        }
        labels = {
            'title': 'Titre de la vidéo',
            'file': 'Fichier vidéo'
        }
        help_texts = {
            'title': 'Maximum 100 caractères',
            'file': 'Formats supportés : MP4, AVI, MOV, etc.'
        } 