import os
import whisper
import logging
import re
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcribe_audio.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def transcribe_audio_files(audio_dir, result_dir):
    # Determine device (CUDA if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if device == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load Whisper model (medium model for balance of speed and accuracy)
    model = whisper.load_model("medium", device=device)
    logging.info(f"Whisper model loaded (medium). Device: {model.device}")
    
    # Regular expression to detect "noisy" transcriptions (screams, sighs, etc.)
    noise_pattern = re.compile(r'^(ahh+|ugh+|ohh+|huh+|eek+|grr+|mmm+|noo+|aahh+|hmmm+|AHHHH+|NOOOO+)\s*[!\-]?$', re.IGNORECASE)
    
    # Walk through all files in the directory
    for root, _, files in os.walk(audio_dir):
        # Sort files by name to process in order (0.wav, 1.wav, ...)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        
        # List to store transcriptions for the current folder
        transcriptions = []
        
        for file_name in files:
            if file_name.endswith('.wav'):
                audio_path = os.path.join(root, file_name)
                try:
                    # Transcribe audio
                    result = model.transcribe(audio_path)
                    transcript_text = result["text"].strip()
                    
                    # Check for "noisy" transcriptions
                    is_noise = bool(noise_pattern.match(transcript_text))
                    
                    # Format text for output
                    if not transcript_text:
                        transcriptions.append(f"{file_name}\n[Silent]\n")
                        logging.info(f"No speech detected in {audio_path}, marked as [Silent].")
                    else:
                        final_text = transcript_text
                        if is_noise:
                            final_text = f"[Possible Noise] {transcript_text}"
                        transcriptions.append(f"{file_name}\n{final_text}\n")
                    
                    logging.info(f"Transcription for {audio_path}: {transcript_text if transcript_text else '[Silent]'}")
                    if is_noise:
                        logging.info(f"Possible noise detected in {audio_path}")
                    logging.info("-" * 50)
                
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
        
        # Save results for the current folder immediately
        if transcriptions:
            # Determine relative path from the input directory
            relative_path = os.path.relpath(root, audio_dir)
            
            # Create path for the output file
            result_subdir = os.path.dirname(os.path.join(result_dir, relative_path))
            Path(result_subdir).mkdir(parents=True, exist_ok=True)
            
            # File name is the last folder name (e.g., vo_plffff_1_000.txt)
            folder_name = os.path.basename(root)
            transcript_file_path = os.path.join(result_subdir, f"{folder_name}.txt")
            
            # Write transcriptions to the file
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(transcriptions))
            
            logging.info(f"Transcriptions for folder {relative_path} saved to {transcript_file_path}")
    
    logging.info("Transcription completed.")

def main():
    # Directory with audio files
    audio_directory = "audio_files"  # Change if the path is different
    result_directory = "result"  # Directory for results
    
    if not os.path.exists(audio_directory):
        logging.error(f"Directory {audio_directory} not found. Please specify a valid path.")
        return
    
    logging.info(f"Starting transcription of audio files from {audio_directory}...")
    transcribe_audio_files(audio_dir=audio_directory, result_dir=result_directory)
    logging.info("Process completed.")
    
    # Pause to keep console open
    input("Press Enter to close the console...")

if __name__ == "__main__":
    main()