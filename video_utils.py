import ffmpeg
import tempfile
import os

def extract_audio(video_path: str) -> str | None:
    """
    Extrait la piste audio d'un fichier vidéo, la convertit en WAV mono, 16kHz
    et retourne le chemin vers le fichier audio extrait.

    Args:
        video_path: Chemin vers le fichier vidéo d'entrée.

    Returns:
        Chemin vers le fichier audio WAV extrait, ou None en cas d'erreur.
    """
    try:
        # Créer un nom de fichier temporaire pour la sortie audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            output_audio_path = tmpfile.name
        
        print(f"Extraction de l'audio de {video_path} vers {output_audio_path}")

        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        print(f"Audio extrait avec succès : {output_audio_path}")
        return output_audio_path

    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8') if e.stderr else "Erreur FFMPEG inconnue"
        print(f"Erreur FFMPEG lors de l'extraction audio de {video_path}: {stderr}")
        # Nettoyer le fichier temporaire en cas d'erreur
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de l'extraction audio de {video_path}: {e}")
        # Nettoyer le fichier temporaire en cas d'erreur
        if 'output_audio_path' in locals() and os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        return None

if __name__ == '__main__':
    # Créez un fichier vidéo de test (par exemple, test_video.mp4) dans le même répertoire
    # ou spécifiez le chemin d'accès à une vidéo existante pour tester cette fonction.
    # test_video_file = "test_video.mp4" 
    # if os.path.exists(test_video_file):
    #     print(f"Test de l'extraction audio pour: {test_video_file}")
    #     audio_file = extract_audio(test_video_file)
    #     if audio_file:
    #         print(f"Audio extrait dans: {audio_file}")
    #         # Vous pouvez ajouter ici un appel à speech_transcriber.transcribe(audio_file) pour tester l'ensemble du flux
    #         # N'oubliez pas de supprimer le fichier audio temporaire après utilisation si nécessaire
    #         # os.remove(audio_file)
    #     else:
    #         print("Échec de l'extraction audio.")
    # else:
    # print(f"Fichier vidéo de test '{test_video_file}' non trouvé. Veuillez le créer ou modifier le chemin.")
    print("Le module video_utils est prêt. Utilisez la fonction extract_audio(video_path).") 