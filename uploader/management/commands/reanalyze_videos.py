from django.core.management.base import BaseCommand
from django.db import transaction
from uploader.models import Video
from uploader.ai_analyzer import AITextAnalyzer
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Re-analyse les vidéos existantes avec l\'IA améliorée'

    def add_arguments(self, parser):
        parser.add_argument(
            '--video-id',
            type=int,
            help='ID spécifique de la vidéo à retraiter'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force le retraitement même si déjà analysé'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Simulation sans modification'
        )

    def handle(self, *args, **options):
        video_id = options.get('video_id')
        force = options.get('force', False)
        dry_run = options.get('dry_run', False)

        # Sélectionner les vidéos à traiter
        if video_id:
            videos = Video.objects.filter(id=video_id)
            if not videos.exists():
                self.stdout.write(
                    self.style.ERROR(f'Vidéo avec ID {video_id} introuvable')
                )
                return
        else:
            if force:
                videos = Video.objects.filter(extracted_text__isnull=False)
            else:
                videos = Video.objects.filter(
                    extracted_text__isnull=False,
                    corrected_text__exact=''
                )

        total_videos = videos.count()
        
        if total_videos == 0:
            self.stdout.write(
                self.style.WARNING('Aucune vidéo à traiter')
            )
            return

        self.stdout.write(
            self.style.SUCCESS(
                f'Traitement de {total_videos} vidéo(s)...'
            )
        )

        if dry_run:
            self.stdout.write(
                self.style.WARNING('MODE SIMULATION - Aucune modification')
            )

        # Initialiser l'analyseur IA
        try:
            analyzer = AITextAnalyzer()
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Erreur initialisation IA: {e}')
            )
            return

        # Traiter chaque vidéo
        success_count = 0
        error_count = 0

        for video in videos:
            try:
                self.stdout.write(f'Traitement: {video.title}...')
                
                if not video.extracted_text:
                    self.stdout.write(
                        self.style.WARNING(f'  ⚠️  Pas de texte OCR pour {video.title}')
                    )
                    continue

                # Afficher le texte original
                self.stdout.write(f'  📝 Texte OCR: "{video.extracted_text[:80]}..."')

                # Analyser avec IA
                result = analyzer.analyze_text(video.extracted_text, video.title)
                
                if result and not dry_run:
                    with transaction.atomic():
                        video.corrected_text = result.get('corrected_text', video.extracted_text)
                        video.category = result.get('category', '')
                        video.subcategory = result.get('subcategory', '')
                        video.keywords = result.get('keywords', [])
                        video.analysis_metadata = result.get('metadata', {})
                        video.save()

                # Afficher les résultats
                if result:
                    self.stdout.write(f'  ✅ Texte corrigé: "{result.get("corrected_text", "")[:80]}..."')
                    self.stdout.write(f'  📂 Catégorie: {result.get("category", "N/A")} > {result.get("subcategory", "N/A")}')
                    self.stdout.write(f'  🏷️  Mots-clés: {", ".join(result.get("keywords", [])[:5])}')
                    
                    confidence = result.get('metadata', {}).get('confidence_score', 0)
                    self.stdout.write(f'  📊 Confiance: {confidence:.2f}')
                    
                    success_count += 1
                else:
                    self.stdout.write(
                        self.style.WARNING(f'  ❌ Échec analyse pour {video.title}')
                    )
                    error_count += 1

            except Exception as e:
                error_count += 1
                self.stdout.write(
                    self.style.ERROR(f'  💥 Erreur pour {video.title}: {e}')
                )

        # Résumé final
        self.stdout.write('\n' + '='*50)
        self.stdout.write(
            self.style.SUCCESS(f'✅ Succès: {success_count} vidéo(s)')
        )
        if error_count > 0:
            self.stdout.write(
                self.style.ERROR(f'❌ Erreurs: {error_count} vidéo(s)')
            )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('Simulation terminée - Utilisez sans --dry-run pour appliquer')
            ) 