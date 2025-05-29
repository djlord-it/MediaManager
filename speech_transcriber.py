import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import ffmpeg
import numpy as np

def load_model():
    """Charge le modèle Whisper et le processeur."""
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    return processor, model

def transcribe(audio_path: str) -> str:
    """Transcrire le fichier audio donné en texte, en traitant par segments."""
    processor, model = load_model()

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        # Tenter de convertir avec ffmpeg si le chargement direct échoue
        try:
            out, _ = (
                ffmpeg
                .input(audio_path)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                .run(capture_stdout=True, capture_stderr=True)
            )
            waveform = torch.from_numpy(np.frombuffer(out, np.int16).copy().astype(np.float32) / 32768.0)
            sample_rate = 16000 # Whisper attend du 16kHz
            # Si la waveform est stéréo, la convertir en mono
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)

        except ffmpeg.Error as e:
            print(f"Erreur FFMPEG: {e.stderr.decode('utf8')}")
            return f"Erreur lors du traitement audio avec FFmpeg: {e}"
        except Exception as e_ffmpeg:
             return f"Erreur inattendue lors de la conversion avec FFmpeg: {e_ffmpeg}"

    # S'assurer que l'audio est mono pour Whisper
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0) # Convertir en mono en prenant la moyenne des canaux
    elif waveform.ndim == 1 and waveform.shape[0] > 16000 * 30 : # Si c'est un long fichier mono déjà chargé
        pass # Pas besoin de faire la moyenne

    # Rééchantillonner si nécessaire (Whisper attend du 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Diviser l'audio en segments de 30 secondes pour un meilleur traitement
    chunk_length = 30 * 16000  # 30 secondes à 16kHz
    waveform_numpy = waveform.squeeze().numpy()
    
    print(f"Durée totale de l'audio : {len(waveform_numpy) / 16000:.2f} secondes")
    
    transcriptions = []
    
    # Traiter par segments
    for i in range(0, len(waveform_numpy), chunk_length):
        chunk = waveform_numpy[i:i + chunk_length]
        
        # Ignorer les segments trop courts (moins de 1 seconde)
        if len(chunk) < 16000:
            continue
            
        print(f"Traitement du segment {i//chunk_length + 1}: {i/16000:.1f}s - {(i+len(chunk))/16000:.1f}s")
        
        # Traiter ce segment
        input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        
        # Générer les IDs des tokens pour ce segment
        predicted_ids = model.generate(input_features)
        
        # Décoder les IDs des tokens en texte
        segment_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if segment_transcription.strip():
            transcriptions.append(segment_transcription.strip())
            print(f"Segment {i//chunk_length + 1}: {segment_transcription[:50]}...")

    # Combiner toutes les transcriptions
    full_transcription = " ".join(transcriptions)
    print(f"Transcription complète : {len(full_transcription)} caractères")
    
    return full_transcription

if __name__ == '__main__':
    # Ceci est un exemple d'utilisation.
    # Remplacez 'audio.mp3' par le chemin vers votre fichier audio.
    # Assurez-vous que FFmpeg est installé et accessible dans votre PATH.
    # test_audio_path = "audio.mp3" # Créez un fichier audio.mp3 pour tester ou utilisez un chemin existant
    # print(f"Tentative de transcription de {test_audio_path}")
    # transcript = transcribe(test_audio_path)
    # print("Transcription:", transcript)
    print("Le module de transcription est prêt. Utilisez la fonction transcribe(audio_path) avec un chemin de fichier audio.") 