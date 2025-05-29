from django.db import models
from .ocr_utils import extract_text_from_video_file
import json
import os
import tempfile
import requests
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Import des modules de transcription audio
import sys
sys.path.append('/Users/jesseelorddushime/Documents/Projects/MediaManager')
from video_utils import extract_audio
from speech_transcriber import transcribe

# Create your models here.

class Video(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='videos/')
    extracted_text = models.TextField(blank=True, help_text="Texte extrait automatiquement par OCR")
    corrected_text = models.TextField(blank=True, help_text="Texte OCR corrigé par IA")
    audio_transcription = models.TextField(blank=True, default='', help_text="Transcription automatique de l'audio parlé")
    corrected_audio_transcription = models.TextField(blank=True, default='', help_text="Transcription audio corrigée par IA")
    has_speech = models.BooleanField(default=True, help_text="Indique si la vidéo contient de la parole détectée")
    speech_metadata = models.JSONField(default=dict, blank=True, help_text="Métadonnées de détection vocale")
    keywords = models.JSONField(default=list, blank=True, help_text="Mots-clés extraits et analysés")
    category = models.CharField(max_length=100, blank=True, help_text="Catégorie principale")
    subcategory = models.CharField(max_length=100, blank=True, help_text="Sous-catégorie")
    analysis_metadata = models.JSONField(default=dict, blank=True, help_text="Métadonnées d'analyse IA")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['category']),
            models.Index(fields=['subcategory']),
            models.Index(fields=['-uploaded_at']),
        ]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # Vérifier si c'est un nouvel objet avec fichier
        is_new = not self.pk
        
        # Initialiser les champs audio si pas définis
        if not hasattr(self, 'audio_transcription') or self.audio_transcription is None:
            self.audio_transcription = ""
        
        if not hasattr(self, 'corrected_audio_transcription') or self.corrected_audio_transcription is None:
            self.corrected_audio_transcription = ""
        
        # Initialiser les champs VAD si pas définis
        if not hasattr(self, 'has_speech') or self.has_speech is None:
            self.has_speech = True
        
        if not hasattr(self, 'speech_metadata') or self.speech_metadata is None:
            self.speech_metadata = {}
        
        # Sauvegarder d'abord l'objet (Django gère automatiquement l'upload vers GCS)
        super().save(*args, **kwargs)
        
        # Si c'est un nouveau fichier, vérifier et forcer l'upload GCS si nécessaire
        if is_new and self.file:
            self._ensure_file_on_gcs()
        
        # Après sauvegarde, faire l'extraction OCR et transcription audio si c'est un nouvel objet
        if is_new and self.file:
            try:
                # Extraction OCR avec le fichier uploadé
                extracted_text = extract_text_from_video_file(self.file)
                
                if extracted_text:
                    self.extracted_text = extracted_text
                
                # Transcription audio - gérer les fichiers GCS et locaux
                try:
                    video_path = self._get_video_file_path()
                    print(f"Extraction audio pour le fichier vidéo : {video_path}")
                    
                    # Extraire l'audio
                    audio_path = extract_audio(video_path)
                    
                    if audio_path:
                        # Détecter d'abord si il y a de la parole
                        from voice_detection import has_speech
                        speech_detected, speech_meta = has_speech(audio_path)
                        
                        print(f"🎙️ Détection vocale: {'Parole détectée' if speech_detected else 'Aucune parole'}")
                        
                        # Stocker les métadonnées de détection vocale
                        self.has_speech = speech_detected
                        self.speech_metadata = speech_meta
                        
                        if speech_detected:
                            # Transcrire seulement si parole détectée
                            transcription = transcribe(audio_path)
                            if transcription:
                                self.audio_transcription = transcription
                                print(f"Transcription réussie : {transcription[:100]}...")
                            else:
                                self.audio_transcription = ""
                        else:
                            # Pas de parole détectée, pas de transcription
                            self.audio_transcription = ""
                            print(f"🚫 Transcription ignorée : aucune parole détectée")
                        
                        # Nettoyer le fichier audio temporaire
                        os.remove(audio_path)
                        print(f"Fichier audio temporaire supprimé : {audio_path}")
                    
                    # Nettoyer le fichier vidéo temporaire si il existe
                    self._cleanup_temp_file()
                    
                except Exception as e:
                    print(f"Erreur lors de la transcription audio pour {self.title}: {e}")
                    self.audio_transcription = ""
                    self.has_speech = False
                    self.speech_metadata = {'error': str(e)}
                    # Nettoyer le fichier vidéo temporaire même en cas d'erreur
                    self._cleanup_temp_file()
                
                # Analyse IA si pas déjà fait
                if not self.corrected_text:
                    self.analyze_with_ai()
                
                # Mise à jour avec les nouvelles données (sans déclencher save() récursif)
                Video.objects.filter(pk=self.pk).update(
                    extracted_text=self.extracted_text,
                    corrected_text=self.corrected_text,
                    audio_transcription=self.audio_transcription,
                    corrected_audio_transcription=self.corrected_audio_transcription,
                    has_speech=self.has_speech,
                    speech_metadata=self.speech_metadata,
                    keywords=self.keywords,
                    category=self.category,
                    subcategory=self.subcategory,
                    analysis_metadata=self.analysis_metadata
                )
                
                # Nettoyer le fichier local après traitement si on utilise GCS
                self._cleanup_local_file_if_on_gcs()
                
            except Exception as e:
                # En cas d'erreur OCR, garder l'objet sans texte
                print(f"Erreur OCR pour {self.title}: {e}")
                # Nettoyer quand même le fichier local
                self._cleanup_local_file_if_on_gcs()
                pass

    def _ensure_file_on_gcs(self):
        """S'assure que le fichier est bien uploadé sur Google Cloud Storage."""
        from django.conf import settings
        
        # Vérifier si on utilise GCS
        if not settings.DEFAULT_FILE_STORAGE.endswith('GoogleCloudStorage'):
            print("ℹ️ Stockage local utilisé, pas d'upload GCS nécessaire")
            return
        
        try:
            # Tester l'accessibilité du fichier sur GCS
            response = requests.head(self.file.url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Fichier déjà accessible sur GCS: {self.file.url}")
                return
        except Exception as e:
            print(f"⚠️ Fichier non accessible sur GCS, upload manuel nécessaire: {e}")
        
        # Upload manuel vers GCS si nécessaire
        try:
            # Si le fichier est stocké localement mais qu'on utilise GCS, forcer l'upload
            if hasattr(self.file, 'path') and os.path.exists(self.file.path):
                print(f"📤 Upload manuel vers GCS...")
                self._upload_to_gcs_manually()
        except (NotImplementedError, AttributeError):
            # Le fichier n'est pas local, on assume qu'il est déjà sur GCS
            pass

    def _upload_to_gcs_manually(self):
        """Upload manuel du fichier vers Google Cloud Storage."""
        try:
            from google.cloud import storage
            from django.conf import settings
            
            # Stocker le chemin local avant upload pour traitement ultérieur
            local_path = self.file.path if hasattr(self.file, 'path') else None
            
            # Initialiser le client GCS
            client = storage.Client(project=settings.GS_PROJECT_ID)
            bucket = client.bucket(settings.GS_BUCKET_NAME)
            
            # Nom du blob sur GCS
            blob_name = self.file.name
            blob = bucket.blob(blob_name)
            
            # Uploader le fichier
            with open(self.file.path, 'rb') as file_obj:
                blob.upload_from_file(file_obj, content_type='video/mp4')
            
            # Rendre le fichier public
            blob.make_public()
            
            print(f"✅ Fichier uploadé manuellement sur GCS: {blob.public_url}")
            
            # Ne pas supprimer le fichier local tout de suite, on en aura besoin pour OCR/audio
            # Il sera supprimé dans la méthode save() après traitement
            
        except Exception as e:
            print(f"❌ Erreur upload manuel GCS: {e}")
            # Ne pas lever d'exception, continuer avec le fichier local

    def _get_video_file_path(self):
        """
        Obtient le chemin du fichier vidéo, gérant à la fois le stockage local et GCS.
        Télécharge temporairement le fichier si nécessaire.
        """
        from django.conf import settings
        from django.core.files.storage import default_storage
        
        try:
            # Si on utilise le stockage local, utiliser le chemin direct
            if hasattr(self.file, 'path'):
                return self.file.path
        except (NotImplementedError, AttributeError):
            # Le fichier est sur GCS, on doit le télécharger temporairement
            pass
        
        # Téléchargement temporaire depuis GCS
        import tempfile
        import requests
        
        try:
            # Créer un fichier temporaire
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)  # Fermer le descripteur de fichier
            
            # Télécharger le fichier depuis GCS
            file_url = self.file.url
            print(f"Téléchargement temporaire depuis : {file_url}")
            
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            # Écrire le contenu dans le fichier temporaire
            with open(temp_path, 'wb') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            
            print(f"Fichier temporaire créé : {temp_path}")
            
            # Stocker le chemin temporaire pour nettoyage ultérieur
            self._temp_video_path = temp_path
            
            return temp_path
            
        except Exception as e:
            print(f"Erreur téléchargement fichier GCS : {e}")
            # Fallback : utiliser l'URL directement (peut ne pas marcher avec ffmpeg)
            return self.file.url

    def _cleanup_temp_file(self):
        """Nettoie le fichier temporaire si il existe."""
        if hasattr(self, '_temp_video_path') and os.path.exists(self._temp_video_path):
            try:
                os.remove(self._temp_video_path)
                print(f"Fichier vidéo temporaire supprimé : {self._temp_video_path}")
                delattr(self, '_temp_video_path')
            except OSError as e:
                print(f"Erreur suppression fichier temporaire : {e}")

    def analyze_with_ai(self):
        """Analyse le texte avec OpenAI pour correction, catégorisation et extraction de mots-clés."""
        from .ai_analyzer import AITextAnalyzer
        
        try:
            analyzer = AITextAnalyzer()
            
            # 1. Correction OCR séparée
            if self.extracted_text:
                print("🔧 Correction OCR avec IA...")
                ocr_result = analyzer.analyze_text(self.extracted_text, self.title)
                if ocr_result:
                    self.corrected_text = ocr_result.get('corrected_text', 'N/A')
                else:
                    self.corrected_text = 'N/A'
            else:
                self.corrected_text = 'N/A'
            
            # 2. Correction audio séparée
            if self.audio_transcription:
                print("🎤 Correction transcription audio avec IA...")
                audio_result = analyzer.analyze_text(self.audio_transcription, self.title)
                if audio_result:
                    self.corrected_audio_transcription = audio_result.get('corrected_text', 'N/A')
                else:
                    self.corrected_audio_transcription = 'N/A'
            else:
                self.corrected_audio_transcription = 'N/A'
            
            # 3. Analyse combinée pour catégorisation et mots-clés
            print("🧠 Analyse combinée pour catégorisation...")
            combined_result = analyzer.analyze_combined_content(
                self.corrected_text if self.corrected_text != 'N/A' else '',
                self.corrected_audio_transcription if self.corrected_audio_transcription != 'N/A' else '',
                self.title
            )
            
            if combined_result:
                self.keywords = combined_result.get('keywords', [])
                self.category = combined_result.get('category', '')
                self.subcategory = combined_result.get('subcategory', '')
                self.analysis_metadata = combined_result.get('metadata', {})
                
        except Exception as e:
            # En cas d'erreur, on garde les textes originaux avec marqueurs N/A
            print(f"❌ Erreur analyse IA: {e}")
            if not self.corrected_text:
                self.corrected_text = self.extracted_text if self.extracted_text else 'N/A'
            if not self.corrected_audio_transcription:
                self.corrected_audio_transcription = self.audio_transcription if self.audio_transcription else 'N/A'
            self.analysis_metadata = {'error': str(e)}

    def get_keywords_display(self):
        """Retourne les mots-clés sous forme de chaîne pour l'affichage."""
        if isinstance(self.keywords, list):
            return ', '.join(self.keywords)
        return str(self.keywords)

    def get_search_text(self):
        """Retourne le texte complet pour la recherche (titre + texte OCR corrigé + transcription audio corrigée + mots-clés)."""
        search_parts = [self.title]
        
        # Ajouter le texte OCR corrigé
        if self.corrected_text and self.corrected_text != 'N/A':
            search_parts.append(self.corrected_text)
        elif self.extracted_text:
            search_parts.append(self.extracted_text)
            
        # Ajouter la transcription audio corrigée
        if self.corrected_audio_transcription and self.corrected_audio_transcription != 'N/A':
            search_parts.append(self.corrected_audio_transcription)
        elif self.audio_transcription:
            search_parts.append(self.audio_transcription)
            
        if self.keywords:
            if isinstance(self.keywords, list):
                search_parts.extend(self.keywords)
            
        if self.category:
            search_parts.append(self.category)
            
        if self.subcategory:
            search_parts.append(self.subcategory)
            
        return ' '.join(search_parts)

    def _cleanup_local_file_if_on_gcs(self):
        """
        Nettoie le fichier local après traitement si on utilise Google Cloud Storage.
        """
        from django.conf import settings
        
        # Vérifier si on utilise GCS
        if not settings.DEFAULT_FILE_STORAGE.endswith('GoogleCloudStorage'):
            return
        
        try:
            # Si le fichier a un chemin local (pas encore dans le cloud), le supprimer
            if hasattr(self.file, 'path') and os.path.exists(self.file.path):
                local_path = self.file.path
                print(f"🧹 Nettoyage du fichier local : {local_path}")
                os.remove(local_path)
                print(f"✅ Fichier local supprimé après traitement : {local_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors du nettoyage du fichier local : {e}")

    def delete(self, *args, **kwargs):
        """
        Suppression personnalisée pour s'assurer que le fichier sur GCS est aussi supprimé.
        """
        try:
            # Supprimer le fichier physique (local ou GCS)
            if self.file:
                try:
                    # Essayer la suppression standard Django
                    self.file.delete(save=False)
                    print(f"✅ Fichier supprimé via Django: {self.file.name}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression Django, tentative manuelle GCS: {e}")
                    
                    # Suppression manuelle GCS en dernier recours
                    try:
                        self._delete_from_gcs_manually()
                    except Exception as gcs_e:
                        print(f"❌ Échec suppression manuelle GCS: {gcs_e}")
        
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression du fichier: {e}")
        
        # Supprimer l'objet de la base de données
        super().delete(*args, **kwargs)

    def _delete_from_gcs_manually(self):
        """
        Suppression manuelle du fichier depuis Google Cloud Storage.
        """
        try:
            from google.cloud import storage
            from django.conf import settings
            
            # Initialiser le client GCS
            client = storage.Client(project=settings.GS_PROJECT_ID)
            bucket = client.bucket(settings.GS_BUCKET_NAME)
            
            # Nom du blob sur GCS
            blob_name = self.file.name
            blob = bucket.blob(blob_name)
            
            # Supprimer le blob
            if blob.exists():
                blob.delete()
                print(f"✅ Fichier supprimé manuellement de GCS: {blob_name}")
            else:
                print(f"ℹ️ Fichier déjà supprimé de GCS: {blob_name}")
                
        except Exception as e:
            print(f"❌ Erreur suppression manuelle GCS: {e}")
            raise
