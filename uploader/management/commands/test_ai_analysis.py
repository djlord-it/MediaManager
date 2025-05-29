from django.core.management.base import BaseCommand
from uploader.models import Video
from uploader.ai_analyzer import AITextAnalyzer
import os

class Command(BaseCommand):
    help = 'Teste l\'analyse IA sur une vidéo spécifique'

    def add_arguments(self, parser):
        parser.add_argument(
            'video_id',
            type=int,
            help='ID de la vidéo à analyser'
        )

    def handle(self, *args, **options):
        video_id = options['video_id']
        
        # Vérifier la clé OpenAI
        if not os.environ.get('OPENAI_API_KEY'):
            self.stdout.write(
                self.style.ERROR('❌ OPENAI_API_KEY non configurée dans l\'environnement')
            )
            self.stdout.write('Ajoutez votre clé: export OPENAI_API_KEY=sk-...')
            return

        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'❌ Vidéo avec ID {video_id} introuvable')
            )
            return

        if not video.extracted_text:
            self.stdout.write(
                self.style.ERROR(f'❌ Aucun texte OCR pour la vidéo "{video.title}"')
            )
            return

        self.stdout.write(f'🎬 Analyse de: {video.title}')
        self.stdout.write('='*60)
        
        # Afficher le texte OCR original
        self.stdout.write(f'📝 Texte OCR original:')
        self.stdout.write(f'   "{video.extracted_text}"')
        self.stdout.write('')

        # Initialiser l'analyseur
        try:
            analyzer = AITextAnalyzer()
            self.stdout.write('🤖 Analyseur IA initialisé')
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Erreur initialisation: {e}')
            )
            return

        # Analyser le texte
        self.stdout.write('🔄 Analyse en cours...')
        try:
            result = analyzer.analyze_text(video.extracted_text, video.title)
            
            if result:
                self.stdout.write('')
                self.stdout.write('✅ RÉSULTATS DE L\'ANALYSE:')
                self.stdout.write('-' * 40)
                
                # Texte séparé
                separated = result['metadata'].get('separated_text', '')
                if separated != video.extracted_text:
                    self.stdout.write(f'🔗 Mots séparés: "{separated}"')
                
                # Texte corrigé
                self.stdout.write(f'📖 Texte corrigé: "{result["corrected_text"]}"')
                
                # Catégorisation
                self.stdout.write(f'📂 Catégorie: {result["category"]} > {result["subcategory"]}')
                
                # Mots-clés
                keywords = result.get('keywords', [])
                self.stdout.write(f'🏷️  Mots-clés: {", ".join(keywords)}')
                
                # Métadonnées
                confidence = result['metadata'].get('confidence_score', 0)
                self.stdout.write(f'📊 Confiance: {confidence:.2f}')
                
                # Étapes de traitement
                steps = result['metadata'].get('processing_steps', [])
                self.stdout.write(f'⚙️  Étapes: {" → ".join(steps)}')
                
                self.stdout.write('')
                self.stdout.write('💾 Voulez-vous sauvegarder ces résultats? (y/N)')
                choice = input().lower().strip()
                
                if choice == 'y' or choice == 'yes':
                    video.corrected_text = result['corrected_text']
                    video.category = result['category']
                    video.subcategory = result['subcategory']
                    video.keywords = result['keywords']
                    video.analysis_metadata = result['metadata']
                    video.save()
                    
                    self.stdout.write(
                        self.style.SUCCESS('✅ Résultats sauvegardés!')
                    )
                else:
                    self.stdout.write('ℹ️  Résultats non sauvegardés')
                    
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Aucun résultat d\'analyse')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Erreur analyse: {e}')
            ) 