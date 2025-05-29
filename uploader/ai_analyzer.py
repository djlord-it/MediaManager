import openai
import re
import json
import heapq
from typing import Dict, List, Optional, Any
from django.conf import settings
import logging
import os

logger = logging.getLogger(__name__)

class AITextAnalyzer:
    """
    Analyseur IA pour traitement post-OCR avec OpenAI.
    Sépare les mots collés, corrige mot par mot, et catégorise.
    """
    
    CATEGORIES = {
        'Technology': ['Software', 'Hardware', 'Programming', 'AI/ML', 'Web Development'],
        'Business': ['Marketing', 'Finance', 'Management', 'Entrepreneurship', 'Sales'],
        'Education': ['Tutorials', 'Academic', 'Training', 'Courses', 'Research'],
        'Entertainment': ['Gaming', 'Movies', 'Music', 'Comedy', 'TV Shows'],
        'Health': ['Fitness', 'Nutrition', 'Mental Health', 'Medical', 'Wellness'],
        'Lifestyle': ['Travel', 'Food', 'Fashion', 'Home', 'Personal Development'],
        'News': ['Politics', 'World News', 'Local News', 'Analysis', 'Breaking News'],
        'Creative': ['Art', 'Design', 'Photography', 'Writing', 'Crafts'],
        'Science': ['Research', 'Experiments', 'Nature', 'Physics', 'Biology'],
        'Other': ['Miscellaneous', 'Unclassified', 'Mixed Content', 'Personal', 'Unknown']
    }

    def __init__(self):
        """Initialise l'analyseur avec la clé OpenAI."""
        try:
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY non configurée")
            # Utilisation directe de la variable d'environnement pour éviter les problèmes
            os.environ['OPENAI_API_KEY'] = api_key
            self.client = openai.OpenAI()
        except Exception as e:
            logger.error(f"Erreur initialisation OpenAI: {e}")
            raise ValueError(f"OPENAI_API_KEY problème: {e}")

    def analyze_text(self, ocr_text: str, title: str = "") -> Optional[Dict[str, Any]]:
        """
        Analyse complète du texte OCR avec OpenAI.
        
        Args:
            ocr_text: Texte brut extrait par OCR
            title: Titre de la vidéo pour contexte
            
        Returns:
            Dict avec texte corrigé, catégorie, mots-clés et métadonnées
        """
        if not ocr_text or not ocr_text.strip():
            return None

        try:
            # 1. Détecter et séparer les mots collés
            separated_text = self._separate_glued_words(ocr_text)
            
            # 2. Corriger chaque mot individuellement 
            corrected_text = self._correct_words_individually(separated_text)
            
            # 3. Catégoriser et extraire mots-clés
            analysis_result = self._categorize_and_extract_keywords(corrected_text, title)
            
            # 4. Compiler le résultat final
            result = {
                'corrected_text': corrected_text,
                'category': analysis_result.get('category', ''),
                'subcategory': analysis_result.get('subcategory', ''),
                'keywords': analysis_result.get('keywords', []),
                'metadata': {
                    'original_text': ocr_text,
                    'separated_text': separated_text,
                    'original_length': len(ocr_text),
                    'corrected_length': len(corrected_text),
                    'confidence_score': analysis_result.get('confidence', 0.8),
                    'processing_steps': ['word_separation', 'individual_correction', 'categorization'],
                    'analysis_summary': f"Texte traité avec {len(analysis_result.get('keywords', []))} mots-clés extraits"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse IA: {e}")
            # Retourner au minimum le texte original
            return {
                'corrected_text': ocr_text,
                'category': 'Other',
                'subcategory': 'Unclassified',
                'keywords': [],
                'metadata': {'error': str(e)}
            }

    def analyze_combined_content(self, ocr_text: str, audio_transcription: str, title: str = "") -> Dict[str, Any]:
        """
        Analyse combinée du texte OCR et de la transcription audio.
        Privilégie la transcription audio si l'OCR est trop corrompu.
        """
        try:
            # Nettoyer et corriger le texte OCR
            corrected_ocr = self._correct_words_individually(ocr_text) if ocr_text else ""
            
            # Évaluer la qualité de l'OCR corrigé
            ocr_quality = self._evaluate_text_quality(corrected_ocr)
            audio_quality = self._evaluate_text_quality(audio_transcription)
            
            print(f"📊 Qualité OCR: {ocr_quality:.2f}, Qualité Audio: {audio_quality:.2f}")
            
            # Déterminer la source principale de contenu
            if audio_quality > 0.7 and ocr_quality < 0.3:
                # Audio de bonne qualité, OCR mauvais -> utiliser uniquement l'audio
                primary_text = audio_transcription
                secondary_text = ""
                print(f"🎤 Utilisation audio seul (OCR trop corrompu)")
            elif ocr_quality > 0.7 and audio_quality < 0.3:
                # OCR de bonne qualité, audio mauvais -> utiliser uniquement l'OCR
                primary_text = corrected_ocr
                secondary_text = ""
                print(f"👁️ Utilisation OCR seul (audio faible)")
            elif audio_quality > 0.5 and self._is_likely_audio_transcription(audio_transcription):
                # Audio cohérent même si pas parfait -> privilégier l'audio
                primary_text = audio_transcription
                secondary_text = corrected_ocr if ocr_quality > 0.3 else ""
                print(f"🎤 Audio privilégié (cohérent)")
            elif ocr_quality > 0.5 and audio_quality > 0.5:
                # Les deux sont de qualité acceptable -> combiner intelligemment
                if self._is_likely_audio_transcription(audio_transcription) and len(audio_transcription) > len(corrected_ocr):
                    primary_text = audio_transcription
                    secondary_text = corrected_ocr
                    print(f"🔄 Combinaison audio (principal) + OCR")
                else:
                    primary_text = corrected_ocr
                    secondary_text = audio_transcription
                    print(f"🔄 Combinaison OCR (principal) + audio")
            else:
                # Qualité faible des deux -> utiliser le meilleur ou combiner
                if audio_quality >= ocr_quality and self._is_likely_audio_transcription(audio_transcription):
                    primary_text = audio_transcription
                    secondary_text = corrected_ocr if corrected_ocr else ""
                    print(f"🎤 Audio privilégié (meilleur des deux)")
                elif ocr_quality > audio_quality:
                    primary_text = corrected_ocr
                    secondary_text = audio_transcription if audio_transcription else ""
                    print(f"👁️ OCR privilégié (meilleur des deux)")
                else:
                    # Scores égaux -> privilégier l'audio s'il semble naturel
                    if self._is_likely_audio_transcription(audio_transcription):
                        primary_text = audio_transcription
                        secondary_text = corrected_ocr if corrected_ocr else ""
                        print(f"🎤 Audio privilégié (naturalité)")
                    else:
                        primary_text = corrected_ocr if corrected_ocr else audio_transcription
                        secondary_text = audio_transcription if corrected_ocr else ""
                        print(f"⚖️ Fallback sur meilleur contenu disponible")
            
            # Créer le prompt en fonction de la stratégie choisie
            if secondary_text:
                content_description = f"""
Contenu principal (prioritaire): {primary_text}

Contenu secondaire (complémentaire): {secondary_text}

Instructions: Analyse le contenu principal en priorité. Utilise le contenu secondaire uniquement pour compléter ou préciser si nécessaire."""
            else:
                content_description = f"Contenu à analyser: {primary_text}"
            
            prompt = f"""
Tu es un expert en analyse de contenu multimédia spécialisé dans les vlogs, travel content et lifestyle videos.

Vidéo intitulée: "{title}"

{content_description}

ANALYSE CONTEXTUELLE IMPORTANTE:
- Si c'est une transcription audio de vlog (ex: "Hi guys, I'm at..."), privilégie le contexte lifestyle/travel
- Si le contenu mentionne des lieux (Dubai, city walk, mall, etc.), c'est probablement du Travel/Lifestyle
- Si c'est de la parole naturelle avec "I", "we", "guys", c'est probablement un vlog personnel
- Si il y a des mentions de shopping, lieux, expériences, c'est du Lifestyle content

Fournis une analyse complète au format JSON avec:
1. "corrected_text": Version propre et lisible du contenu (garde le style parlé naturel pour les vlogs)
2. "keywords": Liste de 6-10 mots-clés pertinents (lieu, activité, style, contexte)
3. "category": Priorité aux catégories Lifestyle, Travel, Entertainment pour les vlogs
4. "subcategory": Travel pour lieux/voyages, Personal Development pour vlogs personnels, Food pour restaurants, etc.
5. "metadata": Objet avec confidence_score (0-100), language (en/fr), sentiment, content_type (vlog/tutorial/etc)

RÈGLES SPÉCIALES VLOGS:
- Pour audio parlé naturel, catégorie = "Lifestyle" ou "Entertainment" 
- Mots-clés doivent inclure: lieu (dubai, city), activité (shopping, exploring), style (vlog, lifestyle)
- Garde les expressions parlées naturelles comme "Hi guys", "I'm at", etc.
- Si mentionné: dubai→dubai, travel→travel, shopping→shopping, explore→exploration

Retourne uniquement le JSON valide:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parser la réponse JSON
            result = self._parse_json_response(result_text)
            
            if result:
                # Ajouter des métadonnées sur la source utilisée
                result['metadata']['source_strategy'] = 'audio_priority' if audio_quality > ocr_quality else 'ocr_priority'
                result['metadata']['ocr_quality'] = round(ocr_quality, 2)
                result['metadata']['audio_quality'] = round(audio_quality, 2)
                result['metadata']['original_ocr_length'] = len(ocr_text) if ocr_text else 0
                result['metadata']['corrected_ocr_length'] = len(corrected_ocr) if corrected_ocr else 0
                result['metadata']['audio_length'] = len(audio_transcription) if audio_transcription else 0
                
                print(f"✅ Analyse combinée réussie - Stratégie: {result['metadata']['source_strategy']}")
                return result
            else:
                print("❌ Échec parsing JSON, fallback sur méthode simple")
                return self.analyze_text(primary_text, title)
                
        except Exception as e:
            logger.error(f"Erreur analyse combinée: {e}")
            print(f"❌ Erreur analyse combinée: {e}")
            # Fallback sur la transcription audio si disponible, sinon OCR
            fallback_text = audio_transcription if audio_transcription else ocr_text
            return self.analyze_text(fallback_text, title)

    def _separate_glued_words(self, text: str) -> str:
        """Sépare les mots collés ensemble en utilisant OpenAI."""
        try:
            prompt = f"""
Tu es un expert en correction OCR. Le texte suivant contient des mots collés ensemble sans espaces.
Ton travail est de séparer UNIQUEMENT les mots collés, sans changer le vocabulaire ou la signification.

Règles:
- Sépare seulement les mots qui sont visiblement collés
- Garde le vocabulaire original de l'utilisateur (argot, abréviations, etc.)
- Ne corrige PAS la grammaire ou l'orthographe ici
- Ajoute des espaces où nécessaire
- Préserve la ponctuation existante

Texte OCR: "{text}"

Texte avec mots séparés:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            separated = response.choices[0].message.content.strip()
            return separated if separated else text
            
        except Exception as e:
            logger.error(f"Erreur séparation mots: {e}")
            # Fallback: séparation basique pour démonstration
            return self._basic_word_separation(text)

    def _basic_word_separation(self, text: str) -> str:
        """Séparation basique des mots collés pour démonstration."""
        # Corrections spécifiques connues
        corrections = {
            'Whenyourealizevou': 'When you realize you',
            'cantteven': "can't even",
            'writeasimple': 'write a simple',
            'loopwithout': 'loop without',
            'POV:Looking': 'POV: Looking',
            'theright': 'the right',
            'worshipsong': 'worship song',
            'thatwill': 'that will'
        }
        
        result = text
        for wrong, correct in corrections.items():
            result = result.replace(wrong, correct)
        
        return result

    def _correct_words_individually(self, text):
        """
        Corrige les mots individuellement en supprimant les erreurs OCR courantes.
        Essaie d'abord la correction OpenAI avant la détection de corruption agressive.
        """
        if not text or text.strip() == "":
            return ""
        
        # Pour les textes avec des mots collés mais reconnaissables, essayer d'abord OpenAI
        if self._looks_like_glued_words(text):
            print(f"🔗 Texte avec mots collés détecté, tentative correction OpenAI...")
            try:
                # Essayer la correction complète avec OpenAI
                prompt = f"""
Corrige ce texte OCR en séparant les mots collés et en corrigeant les erreurs:

Texte: "{text}"

Retourne uniquement le texte corrigé, naturel et lisible:"""

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                
                corrected = response.choices[0].message.content.strip()
                if corrected and len(corrected) > 5:  # Résultat raisonnable
                    print(f"✅ Correction OpenAI réussie: {corrected}")
                    return corrected
                    
            except Exception as e:
                print(f"❌ Erreur correction OpenAI: {e}")
        
        # Détecter si le texte est très corrompu (seulement après échec OpenAI)
        if self._is_heavily_corrupted(text):
            print(f"🗑️ Texte OCR détecté comme très corrompu après tentative correction")
            return ""
        
        # Nettoyage mot par mot pour les autres cas
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Nettoyer la ponctuation pour l'analyse
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Ignorer les mots trop courts ou vides
            if len(clean_word) < 2:
                continue
            
            # Supprimer les mots avec trop de caractères répétés
            if self._has_excessive_repetition(clean_word):
                continue
            
            # Supprimer les mots avec trop de caractères aléatoires
            if self._is_random_characters(clean_word):
                continue
            
            # Garder seulement les mots qui semblent valides
            if self._is_valid_word(clean_word):
                corrected_words.append(word)
        
        result = ' '.join(corrected_words)
        
        # Vérification finale : si le résultat est encore très corrompu, le supprimer
        if self._is_heavily_corrupted(result):
            print(f"🗑️ Résultat encore corrompu après nettoyage, suppression totale")
            return ""
        
        return result

    def _looks_like_glued_words(self, text):
        """Détecte si le texte ressemble à des mots collés ensemble mais reconnaissables."""
        if len(text) < 10:
            return False
            
        # Patterns de mots collés typiques
        glued_patterns = [
            r'[a-z]{4,}[A-Z][a-z]{3,}',  # motMinuscule suivi de MotMajuscule
            r'\b\w{8,}\b',  # Mots très longs (probablement collés)
            r'[a-z]+[a-z]+[a-z]+',  # Séquences sans espaces
        ]
        
        # Vérifier la présence de mots potentiellement corrects
        recognizable_words = [
            'you', 'cant', 'can', 'even', 'write', 'simple', 'loop', 'without', 
            'consulting', 'chatgpt', 'when', 'realize', 'that', 'the', 'and',
            'programming', 'code', 'computer', 'software'
        ]
        
        text_lower = text.lower()
        word_matches = sum(1 for word in recognizable_words if word in text_lower)
        
        # Si on trouve des patterns de mots collés ET des mots reconnaissables
        pattern_matches = sum(1 for pattern in glued_patterns if re.search(pattern, text))
        
        return pattern_matches > 0 and word_matches >= 2

    def _is_heavily_corrupted(self, text):
        """Détecte si un texte est très corrompu avec trop de caractères aléatoires."""
        if not text or len(text.strip()) < 10:
            return False
        
        words = text.split()
        if len(words) < 3:
            return len(text) > 50  # Texte long mais peu de mots = corrompu
        
        # Calculer le ratio de mots corrompus
        corrupted_words = 0
        total_words = len(words)
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) < 2:
                continue
                
            # Vérifier les critères de corruption
            if (self._has_excessive_repetition(clean_word) or 
                self._is_random_characters(clean_word) or 
                not self._is_valid_word(clean_word)):
                corrupted_words += 1
        
        # Si plus de 70% des mots sont corrompus, considérer le texte comme corrompu
        corruption_ratio = corrupted_words / max(total_words, 1)
        return corruption_ratio > 0.7

    def _has_excessive_repetition(self, word):
        """Détecte les mots avec trop de caractères répétés."""
        if len(word) < 3:
            return False
        
        # Compter les caractères répétés consécutifs
        consecutive_count = 1
        for i in range(1, len(word)):
            if word[i] == word[i-1]:
                consecutive_count += 1
                if consecutive_count > 2:  # Plus de 2 caractères identiques consécutifs
                    return True
            else:
                consecutive_count = 1
        
        return False

    def _is_random_characters(self, word):
        """Détecte les séquences de caractères apparemment aléatoires."""
        if len(word) < 3:
            return len(word) == 1 and word.isupper()  # Lettres isolées en majuscules
        
        # Vérifier le ratio voyelles/consonnes (doit être raisonnable)
        vowels = sum(1 for c in word.lower() if c in 'aeiou')
        consonants = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
        
        if consonants == 0:
            return vowels > 3  # Trop de voyelles consécutives
        
        vowel_ratio = vowels / (vowels + consonants)
        
        # Mots avec trop peu ou trop de voyelles sont suspects
        if vowel_ratio < 0.1 or vowel_ratio > 0.8:
            return True
        
        # Vérifier les patterns suspects (trop de majuscules mélangées)
        upper_count = sum(1 for c in word if c.isupper())
        if len(word) > 3 and upper_count > len(word) * 0.6:
            return True
        
        return False

    def _is_valid_word(self, word):
        """Vérifie si un mot semble valide (critères plus stricts)."""
        if len(word) < 2:
            return False
        
        # Rejeter les mots avec uniquement des chiffres ou caractères spéciaux
        if not any(c.isalpha() for c in word):
            return False
        
        # Rejeter les mots avec des patterns très suspects
        suspicious_patterns = [
            r'^[A-Z]{3,}$',  # Trop de majuscules consécutives
            r'[0-9]{3,}',    # Trop de chiffres
            r'^[bcdfghjklmnpqrstvwxyz]{4,}$',  # Trop de consonnes sans voyelles
            r'^[aeiou]{3,}$', # Trop de voyelles consécutives
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return False
        
        return True

    def _categorize_and_extract_keywords(self, text: str, title: str = "") -> Dict[str, Any]:
        """Catégorise le contenu et extrait les mots-clés pertinents."""
        try:
            categories_list = "\n".join([f"- {cat}: {', '.join(subs)}" for cat, subs in self.CATEGORIES.items()])
            
            prompt = f"""
Analyse ce contenu vidéo et extrait les informations suivantes en JSON:

Contenu:
Titre: "{title}"
Texte: "{text}"

Catégories disponibles:
{categories_list}

Retourne un JSON avec:
{{
    "category": "catégorie principale",
    "subcategory": "sous-catégorie spécifique",
    "keywords": ["mot-clé1", "mot-clé2", ...],
    "confidence": 0.95
}}

Règles:
- Choisis la catégorie la plus pertinente
- Extrait 5-10 mots-clés importants et spécifiques
- Évite les mots communs (le, la, de, etc.)
- Confidence entre 0.1 et 1.0
- Mots-clés en minuscules sauf noms propres

JSON:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Tenter de parser le JSON
            try:
                result = json.loads(result_text)
                
                # Validation et nettoyage
                if result.get('category') not in self.CATEGORIES:
                    result['category'] = 'Other'
                
                if result.get('subcategory') not in self.CATEGORIES.get(result['category'], []):
                    result['subcategory'] = self.CATEGORIES[result['category']][0]
                
                # Nettoyer les mots-clés
                keywords = result.get('keywords', [])
                if isinstance(keywords, list):
                    keywords = [kw.lower().strip() for kw in keywords if kw and len(kw.strip()) > 2]
                    result['keywords'] = keywords[:10]  # Max 10 mots-clés
                
                result['confidence'] = max(0.1, min(1.0, float(result.get('confidence', 0.8))))
                
                return result
                
            except json.JSONDecodeError:
                logger.error(f"JSON invalide reçu d'OpenAI: {result_text}")
                return self._fallback_analysis(text, title)
                
        except Exception as e:
            logger.error(f"Erreur catégorisation: {e}")
            return self._fallback_analysis(text, title)

    def _analyze_combined_content_with_ai(self, ocr_text: str, audio_transcription: str, title: str = "") -> Dict[str, Any]:
        """Analyse le contenu combiné (OCR + Audio) avec OpenAI pour une meilleure compréhension."""
        try:
            categories_list = "\n".join([f"- {cat}: {', '.join(subs)}" for cat, subs in self.CATEGORIES.items()])
            
            prompt = f"""
Analyse ce contenu vidéo combiné (texte affiché + audio parlé) et extrait les informations suivantes en JSON:

Titre de la vidéo: "{title}"
Texte affiché (OCR): "{ocr_text or 'Aucun texte affiché'}"
Contenu parlé (Audio): "{audio_transcription or 'Aucun audio transcrit'}"

Catégories disponibles:
{categories_list}

Analyse le CONTEXTE GLOBAL de la vidéo en combinant toutes les sources d'information.
Exemple: Si audio dit "Hi guys, I'm at the city walk" + "Dubai" + "shopping" + "cars" + "zoo", 
alors mots-clés = ["vlog", "dubai", "city walk", "exploration", "travel", "lifestyle", "shopping"]

Retourne un JSON avec:
{{
    "category": "catégorie principale",
    "subcategory": "sous-catégorie spécifique", 
    "keywords": ["mot-clé1", "mot-clé2", ...],
    "confidence": 0.95,
    "content_type": "vlog|tutorial|entertainment|etc"
}}

Règles:
- Analyse le COMPORTEMENT et l'ACTIVITÉ (ex: vlogging, exploration, tutorial)
- Identifie le LIEU/CONTEXTE (ex: dubai, city walk, mall)
- Identifie les OBJETS/SUJETS mentionnés (ex: cars, shorts, zoo)
- Identifie le STYLE/GENRE (ex: vlog, lifestyle, travel)
- Extrait 8-12 mots-clés pertinents et spécifiques
- Évite les mots trop génériques
- Privilégie les termes descriptifs du contenu

JSON:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Tenter de parser le JSON
            try:
                result = json.loads(result_text)
                
                # Validation et nettoyage
                if result.get('category') not in self.CATEGORIES:
                    result['category'] = 'Lifestyle'  # Default plus approprié pour vlogs
                
                if result.get('subcategory') not in self.CATEGORIES.get(result['category'], []):
                    result['subcategory'] = self.CATEGORIES[result['category']][0]
                
                # Nettoyer les mots-clés
                keywords = result.get('keywords', [])
                if isinstance(keywords, list):
                    keywords = [kw.lower().strip() for kw in keywords if kw and len(kw.strip()) > 2]
                    result['keywords'] = keywords[:12]  # Max 12 mots-clés pour analyse combinée
                
                result['confidence'] = max(0.1, min(1.0, float(result.get('confidence', 0.8))))
                
                return result
                
            except json.JSONDecodeError:
                logger.error(f"JSON invalide reçu d'OpenAI pour analyse combinée: {result_text}")
                return self._fallback_combined_analysis(ocr_text, audio_transcription, title)
                
        except Exception as e:
            logger.error(f"Erreur analyse combinée: {e}")
            return self._fallback_combined_analysis(ocr_text, audio_transcription, title)

    def _analyze_audio_only(self, audio_transcription: str, title: str = "") -> Dict[str, Any]:
        """Analyse de la transcription audio uniquement."""
        return self._categorize_and_extract_keywords(audio_transcription, title)

    def _fallback_combined_analysis(self, ocr_text: str, audio_transcription: str, title: str) -> Dict[str, Any]:
        """Analyse de fallback pour le contenu combiné."""
        # Combiner tout le texte disponible
        combined_text = " ".join(filter(None, [title, ocr_text, audio_transcription]))
        
        # Extraction basique de mots-clés
        words = re.findall(r'\b\w+\b', combined_text.lower())
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those', 'what', 'where', 'when', 'how', 'why', 'who', 'which', 'here', 'there', 'now', 'then', 'like', 'just', 'really', 'very', 'much', 'more', 'most', 'some', 'all', 'any', 'many', 'few', 'little', 'big', 'small', 'good', 'bad', 'new', 'old', 'first', 'last', 'long', 'short', 'high', 'low', 'right', 'left', 'next', 'back', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'because', 'since', 'while', 'until', 'unless', 'if', 'though', 'although', 'however', 'therefore', 'thus', 'also', 'too', 'only', 'even', 'still', 'already', 'yet', 'ever', 'never', 'always', 'sometimes', 'often', 'usually', 'generally', 'probably', 'maybe', 'perhaps', 'actually', 'really', 'definitely', 'certainly', 'surely', 'clearly', 'obviously', 'apparently', 'especially', 'particularly', 'mainly', 'mostly', 'rather', 'quite', 'somewhat', 'pretty', 'very', 'really', 'super', 'so', 'too', 'enough'}
        
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        keywords = list(dict.fromkeys(keywords))[:10]  # Enlever doublons, max 10
        
        # Catégorisation intelligente basée sur le contenu
        text_lower = combined_text.lower()
        
        # Mots-clés spécifiques par catégorie
        if any(word in text_lower for word in ['vlog', 'dubai', 'travel', 'exploring', 'city', 'walk']):
            category = 'Lifestyle'
            subcategory = 'Travel'
        elif any(word in text_lower for word in ['code', 'programming', 'software', 'computer', 'tech', 'app', 'website', 'algorithm']):
            category = 'Technology'
            subcategory = 'Programming'
        elif any(word in text_lower for word in ['game', 'gaming', 'play', 'player', 'level', 'score']):
            category = 'Entertainment'
            subcategory = 'Gaming'
        elif any(word in text_lower for word in ['cooking', 'recipe', 'food', 'kitchen', 'meal']):
            category = 'Lifestyle'
            subcategory = 'Food'
        else:
            category = 'Entertainment'
            subcategory = 'Miscellaneous'
        
        return {
            'category': category,
            'subcategory': subcategory,
            'keywords': keywords,
            'confidence': 0.6
        }

    def _fallback_analysis(self, text: str, title: str) -> Dict[str, Any]:
        """Analyse de fallback si OpenAI échoue."""
        # Extraction basique de mots-clés
        words = re.findall(r'\b\w+\b', text.lower())
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        keywords = list(dict.fromkeys(keywords))[:8]  # Enlever doublons, max 8
        
        # Catégorisation basique
        tech_words = ['code', 'programming', 'software', 'computer', 'tech', 'app', 'website', 'algorithm']
        category = 'Technology' if any(word in text.lower() for word in tech_words) else 'Other'
        subcategory = 'Programming' if category == 'Technology' else 'Miscellaneous'
        
        return {
            'category': category,
            'subcategory': subcategory,
            'keywords': keywords,
            'confidence': 0.6
        }

    def _evaluate_text_quality(self, text: str) -> float:
        """
        Évalue la qualité d'un texte (0.0 = très mauvais, 1.0 = excellent).
        Ajusté pour mieux détecter les transcriptions audio vs OCR corrompu.
        """
        if not text or text.strip() == "":
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        score = 1.0
        
        # Détecter si c'est probablement de l'audio (phrases cohérentes)
        is_likely_audio = self._is_likely_audio_transcription(text)
        
        # Détecter si c'est probablement de l'OCR (mots isolés, fragments)
        is_likely_ocr = self._is_likely_ocr_text(text)
        
        print(f"🔍 Analyse qualité: audio={is_likely_audio}, ocr={is_likely_ocr}")
        
        word_count = len(words)
        
        # Scoring différent pour audio vs OCR
        if is_likely_audio:
            # Pour l'audio, privilégier la cohérence même si court
            print(f"🎤 Détecté comme transcription audio")
            
            # Moins pénaliser les transcriptions courtes si elles sont cohérentes
            if word_count < 10:
                score *= 0.7  # Réduction modérée au lieu de 0.5
            elif word_count > 300:  # Transcriptions très longues peuvent être complètes
                score *= 1.1
                
            # Bonus pour structure de phrases naturelles
            if self._has_natural_speech_patterns(text):
                score *= 1.3
                print(f"🗣️ Patterns de parole naturelle détectés")
            
            # Bonus pour ponctuation et structure
            if re.search(r'[.!?]', text):
                score *= 1.1
                
        elif is_likely_ocr:
            # Pour l'OCR, plus strict sur la longueur et cohérence
            print(f"👁️ Détecté comme texte OCR")
            
            if word_count < 3:
                score *= 0.3  # Très pénalisant pour OCR court
            elif word_count > 200:
                score *= 0.8
            
            # Pénaliser davantage l'OCR fragmenté
            if self._is_fragmented_text(text):
                score *= 0.4
                print(f"📝 Texte fragmenté détecté")
        else:
            # Évaluation standard
            if word_count < 3:
                score *= 0.5
            elif word_count > 200:
                score *= 0.8
        
        # Évaluer la corruption du texte (commun pour tous)
        corrupted_words = 0
        valid_words = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) < 2:
                continue
            
            valid_words += 1
            if (self._has_excessive_repetition(clean_word) or 
                self._is_random_characters(clean_word) or 
                not self._is_valid_word(clean_word)):
                corrupted_words += 1
        
        if valid_words > 0:
            corruption_ratio = corrupted_words / valid_words
            score *= (1.0 - corruption_ratio)
            print(f"📊 Ratio corruption: {corruption_ratio:.2f}")
        
        # Vérifier la cohérence linguistique
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio > 0.7:
            score *= 1.1
        elif alpha_ratio < 0.3:
            score *= 0.5
        
        final_score = min(score, 1.0)
        print(f"📊 Score final: {final_score:.2f}")
        return final_score

    def _is_likely_audio_transcription(self, text: str) -> bool:
        """Détecte si le texte ressemble à une transcription audio."""
        # Patterns typiques de parole
        speech_indicators = [
            r'\b(hi|hey|hello|guys?|everyone)\b',  # Salutations
            r'\b(i\'m|i am|we\'re|we are)\b',      # Contractions courantes
            r'\b(gonna|wanna|gotta)\b',            # Langage parlé informel
            r'\b(like|you know|actually|really)\b', # Mots de remplissage
            r'\b(what|where|how|why)\s+\w+',       # Questions naturelles
            r'\b(let\'s|let me|look at)\b',        # Actions parlées
            r'\b(this is|that is|here is|there is)\b' # Démonstratifs
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in speech_indicators if re.search(pattern, text_lower))
        
        # Vérifier la structure de phrase
        has_pronouns = bool(re.search(r'\b(i|you|we|they|he|she|it)\b', text_lower))
        has_verbs = bool(re.search(r'\b(am|is|are|was|were|have|has|do|does|did|can|will|would)\b', text_lower))
        
        return matches >= 2 or (has_pronouns and has_verbs)

    def _is_likely_ocr_text(self, text: str) -> bool:
        """Détecte si le texte ressemble à de l'OCR (titres, captions, fragments)."""
        # Patterns typiques d'OCR
        ocr_indicators = [
            r'^[A-Z][A-Z\s]+$',                    # TITRES EN MAJUSCULES
            r'\b\d+[%$€£]\b',                      # Chiffres avec symboles
            r'\b[A-Z]{2,}\b',                      # Acronymes multiples
            r'^[^.!?]*$',                          # Pas de ponctuation de fin
            r'\b(step|chapter|part|section)\s+\d+\b', # Numérotation
            r'\b(click|press|select|choose)\b',    # Instructions UI
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in ocr_indicators if re.search(pattern, text))
        
        # Vérifier fragments courts sans structure
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # OCR tend à avoir des mots plus courts et fragmentés
        is_fragmented = len(words) < 10 and avg_word_length < 4
        
        return matches >= 1 or is_fragmented

    def _has_natural_speech_patterns(self, text: str) -> bool:
        """Vérifie les patterns de parole naturelle."""
        patterns = [
            r'\b(um|uh|er|ah)\b',                  # Hésitations
            r'\b(and|but|so|then|now)\s+',         # Connecteurs de parole
            r'\w+,\s+\w+',                         # Virgules dans le discours
            r'\b(i mean|you see|you know)\b',      # Expressions de parole
            r'\?\s+',                              # Questions
            r'!\s+',                               # Exclamations
        ]
        
        text_lower = text.lower()
        return sum(1 for pattern in patterns if re.search(pattern, text_lower)) >= 2

    def _is_fragmented_text(self, text: str) -> bool:
        """Détecte si le texte est fragmenté (typique de l'OCR défaillant)."""
        words = text.split()
        if len(words) < 5:
            return True
        
        # Vérifier la cohérence entre mots adjacents
        fragments = 0
        for i in range(len(words) - 1):
            word1 = re.sub(r'[^\w]', '', words[i].lower())
            word2 = re.sub(r'[^\w]', '', words[i + 1].lower())
            
            # Mots très courts successifs = fragmentation
            if len(word1) <= 2 and len(word2) <= 2:
                fragments += 1
        
        fragmentation_ratio = fragments / max(len(words) - 1, 1)
        return fragmentation_ratio > 0.3

    def _parse_json_response(self, result_text: str) -> Dict[str, Any]:
        """Parse la réponse JSON et retourne le résultat."""
        try:
            result = json.loads(result_text)
            
            # Validation et nettoyage
            if result.get('category') not in self.CATEGORIES:
                result['category'] = 'Other'
            
            if result.get('subcategory') not in self.CATEGORIES.get(result['category'], []):
                result['subcategory'] = self.CATEGORIES[result['category']][0]
            
            # Nettoyer les mots-clés
            keywords = result.get('keywords', [])
            if isinstance(keywords, list):
                keywords = [kw.lower().strip() for kw in keywords if kw and len(kw.strip()) > 2]
                result['keywords'] = keywords[:10]  # Max 10 mots-clés
            
            result['confidence'] = max(0.1, min(1.0, float(result.get('confidence', 0.8))))
            
            return result
            
        except json.JSONDecodeError:
            logger.error(f"JSON invalide reçu d'OpenAI: {result_text}")
            return None


class SmartSearch:
    """Système de recherche intelligent avec scoring par heap."""
    
    @staticmethod
    def search_videos(query: str, queryset=None) -> List:
        """
        Recherche intelligente avec scoring par pertinence.
        
        Args:
            query: Terme de recherche
            queryset: QuerySet de vidéos à filtrer (optionnel)
            
        Returns:
            Liste de vidéos triées par pertinence
        """
        from .models import Video
        
        if queryset is None:
            videos = Video.objects.all()
        else:
            videos = queryset
            
        if not query:
            return list(videos.order_by('-uploaded_at'))
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Heap pour stocker (score_negatif, index, video) - index évite la comparaison directe des objets
        scored_videos = []
        
        for index, video in enumerate(videos):
            score = SmartSearch._calculate_relevance_score(video, query_lower, query_words)
            if score > 0:
                # Score négatif pour avoir les meilleurs scores en premier avec heapq
                # Index ajouté pour éviter la comparaison directe des objets Video
                heapq.heappush(scored_videos, (-score, index, video))
        
        # Extraire les vidéos triées par score décroissant
        sorted_videos = []
        while scored_videos:
            neg_score, index, video = heapq.heappop(scored_videos)
            sorted_videos.append(video)
        
        return sorted_videos
    
    @staticmethod
    def _calculate_relevance_score(video, query_lower: str, query_words: List[str]) -> float:
        """Calcule le score de pertinence d'une vidéo pour la requête."""
        score = 0.0
        
        # Poids par champ
        weights = {
            'title': 3.0,
            'category': 2.5,
            'subcategory': 2.0,
            'keywords': 2.0,
            'corrected_text': 1.0,
            'extracted_text': 0.8
        }
        
        # Recherche dans le titre
        if video.title and query_lower in video.title.lower():
            score += weights['title']
        
        # Recherche dans la catégorie
        if video.category and query_lower in video.category.lower():
            score += weights['category']
            
        # Recherche dans la sous-catégorie
        if video.subcategory and query_lower in video.subcategory.lower():
            score += weights['subcategory']
        
        # Recherche dans les mots-clés
        if video.keywords and isinstance(video.keywords, list):
            keyword_matches = sum(1 for kw in video.keywords if kw and query_lower in kw.lower())
            score += keyword_matches * weights['keywords']
        
        # Recherche dans le texte corrigé
        if video.corrected_text and query_lower in video.corrected_text.lower():
            score += weights['corrected_text']
        
        # Recherche dans le texte extrait (fallback)
        if video.extracted_text and query_lower in video.extracted_text.lower():
            score += weights['extracted_text']
        
        # Bonus pour correspondance de mots multiples
        for word in query_words:
            if len(word) > 2:  # Ignorer les mots trop courts
                text_to_search = f"{video.title} {video.corrected_text or video.extracted_text} {' '.join(video.keywords or [])}".lower()
                if word in text_to_search:
                    score += 0.5
        
        return score 