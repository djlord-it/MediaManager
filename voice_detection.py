#!/usr/bin/env python3
"""
Module de détection d'activité vocale (VAD) utilisant Silero VAD.
Détecte si un fichier audio contient de la parole humaine.
"""

import torch
import torchaudio
import os
import tempfile
from typing import Tuple, Optional
import warnings

# Supprimer les warnings de torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

class VoiceActivityDetector:
    """Détecteur d'activité vocale utilisant Silero VAD."""
    
    def __init__(self):
        """Initialise le détecteur VAD."""
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle Silero VAD."""
        try:
            import silero_vad
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            print("✅ Modèle Silero VAD chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur chargement modèle VAD: {e}")
            raise

    def has_speech(self, audio_path: str, min_speech_duration: float = 1.0, 
                   confidence_threshold: float = 0.5) -> Tuple[bool, dict]:
        """
        Détecte si le fichier audio contient de la parole humaine.
        
        Args:
            audio_path: Chemin vers le fichier audio
            min_speech_duration: Durée minimale de parole requise (secondes)
            confidence_threshold: Seuil de confiance pour la détection
            
        Returns:
            Tuple (has_speech: bool, metadata: dict)
        """
        try:
            # Charger l'audio
            wav, sample_rate = torchaudio.load(audio_path)
            
            # Convertir en mono si nécessaire
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample à 16kHz si nécessaire (requis par Silero VAD)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                wav = resampler(wav)
                sample_rate = 16000
            
            # Appliquer VAD
            speech_timestamps = self._get_speech_timestamps(wav, sample_rate)
            
            # Calculer les métriques
            total_speech_duration = sum(
                (end - start) / sample_rate 
                for start, end in speech_timestamps
            )
            
            audio_duration = wav.shape[1] / sample_rate
            speech_ratio = total_speech_duration / max(audio_duration, 0.1)
            
            # Déterminer si il y a de la parole
            has_speech = (
                total_speech_duration >= min_speech_duration and 
                speech_ratio >= 0.1  # Au moins 10% de parole
            )
            
            metadata = {
                'total_duration': round(audio_duration, 2),
                'speech_duration': round(total_speech_duration, 2),
                'speech_ratio': round(speech_ratio, 3),
                'speech_segments': len(speech_timestamps),
                'confidence': confidence_threshold,
                'method': 'silero_vad'
            }
            
            print(f"🎙️ VAD: Parole={has_speech}, Durée={total_speech_duration:.1f}s/{audio_duration:.1f}s ({speech_ratio:.1%})")
            
            return has_speech, metadata
            
        except Exception as e:
            print(f"❌ Erreur détection vocale: {e}")
            # En cas d'erreur, on assume qu'il y a de la parole (sécurité)
            return True, {'error': str(e), 'fallback': True}

    def _get_speech_timestamps(self, wav: torch.Tensor, sample_rate: int) -> list:
        """Obtient les timestamps des segments de parole."""
        try:
            # Utiliser get_speech_timestamps de Silero VAD
            speech_timestamps = self.utils[0](
                wav.squeeze(), 
                self.model, 
                sampling_rate=sample_rate,
                threshold=0.5,
                min_speech_duration_ms=250,  # 250ms minimum
                min_silence_duration_ms=100,  # 100ms de silence entre segments
                window_size_samples=1536,
                speech_pad_ms=30
            )
            
            # Convertir en liste de tuples (start, end)
            return [(segment['start'], segment['end']) for segment in speech_timestamps]
            
        except Exception as e:
            print(f"❌ Erreur get_speech_timestamps: {e}")
            return []

def has_speech(audio_path: str) -> Tuple[bool, dict]:
    """
    Fonction helper pour détecter la parole dans un fichier audio.
    
    Args:
        audio_path: Chemin vers le fichier audio
        
    Returns:
        Tuple (has_speech: bool, metadata: dict)
    """
    detector = VoiceActivityDetector()
    return detector.has_speech(audio_path)

def test_voice_detection(audio_path: str):
    """Teste la détection vocale sur un fichier."""
    print(f"🔍 Test détection vocale: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"❌ Fichier introuvable: {audio_path}")
        return
    
    has_speech_result, metadata = has_speech(audio_path)
    
    print(f"📊 Résultats:")
    print(f"   Parole détectée: {'✅ OUI' if has_speech_result else '❌ NON'}")
    print(f"   Durée totale: {metadata.get('total_duration', 'N/A')}s")
    print(f"   Durée parole: {metadata.get('speech_duration', 'N/A')}s")
    print(f"   Ratio parole: {metadata.get('speech_ratio', 'N/A')}")
    print(f"   Segments: {metadata.get('speech_segments', 'N/A')}")

if __name__ == "__main__":
    # Test avec un fichier audio
    import sys
    if len(sys.argv) > 1:
        test_voice_detection(sys.argv[1])
    else:
        print("Usage: python voice_detection.py <audio_file>") 