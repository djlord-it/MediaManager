import cv2
import pytesseract
import tempfile
import os
import logging
import re
import string

logger = logging.getLogger(__name__)

# Import EasyOCR avec gestion d'erreur
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # Initialiser le reader une seule fois (cache global)
    _easyocr_reader = None
except ImportError:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None

def get_easyocr_reader():
    """Obtient l'instance EasyOCR reader (cache global)."""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

def extract_text_from_video_file(file_field):
    """
    Extrait le texte d'un fichier vidéo avec approche hybride EasyOCR + Tesseract.
    
    Args:
        file_field: Django FileField contenant le fichier vidéo
        
    Returns:
        str: Texte extrait de la vidéo, ou chaîne vide si échec
    """
    if not file_field:
        return ""
    
    tmp_path = None
    try:
        # Créer un fichier temporaire pour la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            for chunk in file_field.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Ouvrir la vidéo avec OpenCV
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            logger.warning(f"Impossible d'ouvrir le fichier vidéo: {file_field.name}")
            return ""
        
        # Obtenir les propriétés de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Échantillonner plusieurs frames (10 premières secondes max)
        max_frames_to_check = min(15, int(fps * 10)) if fps > 0 else 15
        best_frames = sample_best_frames(cap, max_frames_to_check)
        
        cap.release()
        
        if not best_frames:
            logger.warning(f"Aucune frame utilisable trouvée: {file_field.name}")
            return ""
        
        # Tester différentes approches sur les meilleures frames
        all_results = []
        
        for frame in best_frames:
            # Approche 1: EasyOCR (deep learning)
            if EASYOCR_AVAILABLE:
                easyocr_text = extract_with_easyocr(frame)
                if easyocr_text:
                    all_results.append({
                        'text': easyocr_text,
                        'confidence': calculate_text_confidence(easyocr_text),
                        'method': 'EasyOCR'
                    })
            
            # Approche 2: Tesseract avec preprocessing optimisé
            tesseract_text = extract_with_tesseract_enhanced(frame)
            if tesseract_text:
                all_results.append({
                    'text': tesseract_text,
                    'confidence': calculate_text_confidence(tesseract_text),
                    'method': 'Tesseract'
                })
            
            # Approche 3: Preprocessing agressif + Tesseract
            aggressive_text = extract_with_aggressive_preprocessing(frame)
            if aggressive_text:
                all_results.append({
                    'text': aggressive_text,
                    'confidence': calculate_text_confidence(aggressive_text),
                    'method': 'Aggressive'
                })
        
        # Fusionner et choisir le meilleur résultat
        best_text = merge_and_select_best_result(all_results)
        
        # Post-traitement final
        cleaned_text = post_process_text(best_text)
        
        # Calculer la confiance finale
        final_confidence = calculate_text_confidence(cleaned_text)
        
        logger.info(f"Texte extrait de {file_field.name}: {len(cleaned_text)} caractères (confiance: {final_confidence:.2f})")
        return cleaned_text

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction OCR pour {file_field.name}: {str(e)}")
        return ""
    
    finally:
        # Nettoyer le fichier temporaire
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Impossible de supprimer le fichier temporaire: {tmp_path}")

def sample_best_frames(cap, max_frames):
    """
    Échantillonne les meilleures frames basées sur la netteté.
    """
    frames = []
    frame_scores = []
    
    for i in range(max_frames):
        # Échantillonner à différents points dans la vidéo
        frame_pos = i * 15  # Toutes les 15 frames (~0.5 sec à 30fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculer la netteté (variance du Laplacien)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        frames.append(frame)
        frame_scores.append(sharpness)
    
    if not frames:
        return []
    
    # Garder les 5 frames les plus nettes
    sorted_indices = sorted(range(len(frame_scores)), key=lambda i: frame_scores[i], reverse=True)
    best_frames = [frames[i] for i in sorted_indices[:5]]
    
    return best_frames

def extract_with_easyocr(frame):
    """
    Extrait le texte avec EasyOCR (deep learning).
    """
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return ""
        
        # EasyOCR fonctionne mieux avec l'image originale
        results = reader.readtext(frame, detail=0, paragraph=True)
        
        if results:
            # Joindre tous les résultats
            text = ' '.join(results)
            return text.strip()
        
        return ""
    except Exception as e:
        logger.debug(f"Erreur EasyOCR: {e}")
        return ""

def extract_with_tesseract_enhanced(frame):
    """
    Extrait le texte avec Tesseract et preprocessing optimisé.
    """
    try:
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sur-échantillonner
        h, w = gray.shape
        if h < 800:
            scale_factor = 800 / h
            new_w = int(w * scale_factor)
            gray = cv2.resize(gray, (new_w, 800), interpolation=cv2.INTER_CUBIC)
        
        # Preprocessing optimisé
        denoised = cv2.medianBlur(gray, 3)
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 5
        )
        
        # Configuration Tesseract optimale
        config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, lang='eng', config=config)
        
        return text.strip()
    except Exception as e:
        logger.debug(f"Erreur Tesseract: {e}")
        return ""

def extract_with_aggressive_preprocessing(frame):
    """
    Extrait le texte avec preprocessing très agressif pour les cas difficiles.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sur-échantillonnage important
        h, w = gray.shape
        scale_factor = 1200 / h if h < 1200 else 1.5
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Égalisation d'histogramme
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(scaled)
        
        # Débruitage agressif
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Seuillage OTSU
        _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tesseract avec paramètres spéciaux
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
        text = pytesseract.image_to_string(binary, lang='eng', config=config)
        
        return text.strip()
    except Exception as e:
        logger.debug(f"Erreur preprocessing agressif: {e}")
        return ""

def merge_and_select_best_result(results):
    """
    Fusionne les résultats et sélectionne le meilleur.
    """
    if not results:
        return ""
    
    # Trier par confiance
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Si on a un résultat avec une très haute confiance, le prendre
    if results[0]['confidence'] > 0.8:
        return results[0]['text']
    
    # Sinon, essayer de fusionner les meilleurs résultats
    best_texts = [r['text'] for r in results[:3]]
    
    # Trouver le texte le plus complet (le plus long avec des mots valides)
    best_text = ""
    best_score = 0
    
    for text in best_texts:
        # Score basé sur longueur + mots reconnaissables
        words = text.split()
        valid_words = [w for w in words if len(w) > 2 and w.isalpha()]
        score = len(text) + len(valid_words) * 10
        
        if score > best_score:
            best_text = text
            best_score = score
    
    return best_text

def calculate_text_confidence(text):
    """
    Calcule un score de confiance pour le texte extrait.
    """
    if not text:
        return 0
    
    # Critères de qualité
    word_count = len(text.split())
    char_count = len(text)
    
    # Ratio de lettres vs caractères spéciaux
    letters = sum(c.isalpha() for c in text)
    letter_ratio = letters / char_count if char_count > 0 else 0
    
    # Présence de mots anglais communs
    common_words = {'the', 'and', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'for', 'of', 'going', 'after', 'day', 'me', 'you', 'he', 'she', 'it'}
    words = text.lower().split()
    common_count = sum(1 for word in words if word in common_words)
    common_ratio = common_count / word_count if word_count > 0 else 0
    
    # Longueur raisonnable
    length_score = min(char_count / 50, 1.0)  # Optimal ~50 chars
    
    # Score composite
    score = (
        length_score * 0.3 +           # Longueur raisonnable
        letter_ratio * 0.4 +           # Ratio de lettres
        common_ratio * 0.3             # Mots communs
    )
    
    return min(score, 1.0)

def post_process_text(text):
    """
    Post-traitement pour nettoyer le texte extrait.
    """
    if not text:
        return ""
    
    # Remplacer les sauts de ligne par des espaces
    text = re.sub(r'\n+', ' ', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Corrections communes d'OCR
    ocr_corrections = {
        r'\b0\b': 'O',  # 0 -> O
        r'\b1\b': 'I',  # 1 -> I dans certains contextes
        r'\bl\b': 'I',  # l minuscule -> I
        r'\brn\b': 'm', # rn -> m
        r'\bvv\b': 'w', # vv -> w
    }
    
    for pattern, replacement in ocr_corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Supprimer la ponctuation isolée
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    
    # Supprimer la ponctuation en début/fin
    text = text.strip(string.punctuation + ' ')
    
    return text 