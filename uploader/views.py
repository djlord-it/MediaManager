from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Q
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .models import Video
from .forms import VideoUploadForm
from .ai_analyzer import SmartSearch

# Create your views here.

def video_list(request):
    """
    Vue pour afficher la liste des vidéos avec recherche intelligente.
    """
    query = request.GET.get('q', '')
    category_filter = request.GET.get('category', '')
    
    # Base queryset
    videos = Video.objects.all()
    
    # Filtrage par catégorie
    if category_filter and category_filter != 'all':
        videos = videos.filter(category=category_filter)
    
    # Recherche intelligente avec heap si query fournie
    if query:
        # Utiliser la recherche intelligente
        videos_list = SmartSearch.search_videos(query, videos)
        total_videos = len(videos_list)
    else:
        # Liste normale triée par date
        videos_list = list(videos.order_by('-uploaded_at'))
        total_videos = videos.count()
    
    # Pagination sur la liste filtrée
    paginator = Paginator(videos_list, 9)  # 9 vidéos par page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Récupérer les catégories disponibles pour le filtre
    categories = Video.objects.exclude(category='').values_list('category', flat=True).distinct()
    
    context = {
        'page_obj': page_obj,
        'query': query,
        'category_filter': category_filter,
        'categories': categories,
        'total_videos': total_videos,
        'smart_search': bool(query),  # Indication si recherche intelligente utilisée
    }
    return render(request, 'uploader/video_list.html', context)

def video_upload(request):
    """
    Vue pour uploader une nouvelle vidéo avec analyse IA automatique.
    """
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            
            # Message de succès avec informations IA
            success_msg = f'Vidéo "{video.title}" uploadée avec succès !'
            
            if video.corrected_text:
                success_msg += f' Texte extrait et corrigé par IA.'
            if video.category:
                success_msg += f' Catégorisée: {video.category}'
                if video.subcategory:
                    success_msg += f' > {video.subcategory}'
            if video.keywords:
                keyword_count = len(video.keywords) if isinstance(video.keywords, list) else 0
                success_msg += f' {keyword_count} mots-clés extraits.'
            
            messages.success(request, success_msg)
            return redirect('uploader:video_list')
        else:
            messages.error(request, 'Erreur lors de l\'upload. Veuillez vérifier les données.')
    else:
        form = VideoUploadForm()
    
    # Récupérer les 3 dernières vidéos avec leurs métadonnées
    recent_videos = Video.objects.all().order_by('-uploaded_at')[:3]
    
    # Statistiques pour la page d'upload
    stats = {
        'total_videos': Video.objects.count(),
        'categories_count': Video.objects.exclude(category='').values('category').distinct().count(),
        'avg_keywords': 0
    }
    
    # Calculer moyenne mots-clés
    videos_with_keywords = Video.objects.exclude(keywords=[])
    if videos_with_keywords.exists():
        total_keywords = sum(len(v.keywords) for v in videos_with_keywords if isinstance(v.keywords, list))
        stats['avg_keywords'] = round(total_keywords / videos_with_keywords.count(), 1)
    
    context = {
        'form': form,
        'recent_videos': recent_videos,
        'stats': stats,
    }
    return render(request, 'uploader/video_upload.html', context)

def video_detail(request, pk):
    """
    Vue pour afficher les détails complets d'une vidéo avec analyse IA.
    """
    video = get_object_or_404(Video, pk=pk)
    
    # Extraire les mots-clés pour la recherche similaire
    keywords = []
    if video.keywords and isinstance(video.keywords, list):
        keywords = video.keywords[:8]  # Limiter à 8 mots-clés
    elif video.corrected_text:
        # Fallback sur extraction basique si pas de mots-clés IA
        words = video.corrected_text.split()[:8]
        keywords = [word.strip('.,!?;:()[]"\'') for word in words if len(word.strip('.,!?;:()[]"\'')) > 3]
    
    # Rechercher des vidéos similaires
    similar_videos = []
    if video.category and video.category != 'Other':
        similar_videos = Video.objects.filter(
            category=video.category
        ).exclude(pk=video.pk).order_by('-uploaded_at')[:3]
    
    # Métadonnées d'analyse pour l'affichage
    analysis_info = {}
    if video.analysis_metadata:
        analysis_info = {
            'confidence': video.analysis_metadata.get('confidence_score', 0),
            'language': video.analysis_metadata.get('language', 'Unknown'),
            'sentiment': video.analysis_metadata.get('sentiment', 'neutral'),
            'complexity': video.analysis_metadata.get('complexity', 'medium'),
            'summary': video.analysis_metadata.get('analysis_summary', ''),
            'corrections_made': video.analysis_metadata.get('corrected_length', 0) != video.analysis_metadata.get('original_length', 0)
        }
    
    context = {
        'video': video,
        'keywords': keywords,
        'similar_videos': similar_videos,
        'analysis_info': analysis_info,
    }
    return render(request, 'uploader/video_detail.html', context)

def category_view(request, category):
    """
    Vue pour afficher toutes les vidéos d'une catégorie donnée.
    """
    videos = Video.objects.filter(category=category).order_by('-uploaded_at')
    
    # Sous-catégories disponibles
    subcategories = videos.exclude(subcategory='').values_list('subcategory', flat=True).distinct()
    
    # Pagination
    paginator = Paginator(videos, 12)  # 12 vidéos par page pour les catégories
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'category': category,
        'page_obj': page_obj,
        'subcategories': subcategories,
        'total_videos': videos.count(),
    }
    return render(request, 'uploader/category_view.html', context)

def video_delete(request, pk):
    """
    Vue pour supprimer une vidéo avec confirmation.
    """
    video = get_object_or_404(Video, pk=pk)
    
    if request.method == 'POST':
        try:
            video_title = video.title
            video_file_name = video.file.name
            
            # Supprimer le fichier sur Google Cloud Storage si nécessaire
            if video.file:
                try:
                    # Tenter de supprimer le fichier
                    video.file.delete(save=False)
                    print(f"✅ Fichier supprimé: {video_file_name}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression fichier {video_file_name}: {e}")
                    # Continuer la suppression même si le fichier ne peut pas être supprimé
            
            # Supprimer l'entrée de la base de données
            video.delete()
            
            messages.success(request, f'Vidéo "{video_title}" supprimée avec succès.')
            return redirect('uploader:video_list')
            
        except Exception as e:
            messages.error(request, f'Erreur lors de la suppression: {str(e)}')
            return redirect('uploader:video_detail', pk=pk)
    
    # GET request - afficher la page de confirmation
    context = {
        'video': video,
    }
    return render(request, 'uploader/video_delete.html', context)

@require_POST
def video_delete_ajax(request, pk):
    """
    Vue AJAX pour supprimer une vidéo sans rechargement de page.
    """
    try:
        video = get_object_or_404(Video, pk=pk)
        video_title = video.title
        video_file_name = video.file.name
        
        # Supprimer le fichier sur Google Cloud Storage si nécessaire
        if video.file:
            try:
                video.file.delete(save=False)
                print(f"✅ Fichier supprimé: {video_file_name}")
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier {video_file_name}: {e}")
        
        # Supprimer l'entrée de la base de données
        video.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Vidéo "{video_title}" supprimée avec succès.'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Erreur lors de la suppression: {str(e)}'
        })
