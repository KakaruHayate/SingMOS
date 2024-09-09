import os
import torch
import librosa
import pandas as pd
import argparse
from singmos.ssl_mos.ssl_mos import MOS_Predictor, load_ssl_model_s3prl

def load_mos_model(use_cuda):
    model_type = "wav2vec2_base_960"
    ft_model_path = "checkpoints/ft_wav2vec2_base_960_23steps.pt"
    ssl_model, ssl_dim = load_ssl_model_s3prl(model_type)
    model = MOS_Predictor(
            ssl_model_type=model_type,
        )
    if use_cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(torch.load(ft_model_path))
    return model

def predict_mos_for_folder(folder_path, use_cuda=False):
    predictor = load_mos_model(use_cuda)
    
    results = []
    supported_formats = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    files = [f for f in os.listdir(folder_path) if f.endswith(supported_formats)]
    total_files = len(files)
    
    for idx, filename in enumerate(files, 1):
        print(f"Processing {idx}/{total_files}: {filename}")
        
        file_path = os.path.join(folder_path, filename)
        wave, sr = librosa.load(file_path, sr=None, mono=True)
        
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
        
        wave_tensor = torch.from_numpy(wave) # .unsqueeze(0)
        length = torch.tensor([wave_tensor.shape[1]])
        if use_cuda:
            wave_tensor = wave_tensor.cuda()
        
        score = predictor(wave_tensor, length).item()
        results.append([filename, score])
        print(f"Score for {filename}: {score}")

    df = pd.DataFrame(results, columns=["Audio File", "MOS Score"])
    csv_path = os.path.join(folder_path, "mos_scores.csv")
    df.to_csv(csv_path, index=False)

    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict MOS scores for audio files in a folder.')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing .audio files')
    
    args = parser.parse_args()
    folder_path = args.path
    use_cuda = torch.cuda.is_available()
    
    csv_path = predict_mos_for_folder(folder_path, use_cuda=use_cuda)
    print(f"MOS scores saved to {csv_path}")
