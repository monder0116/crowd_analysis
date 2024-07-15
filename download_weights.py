import gdown

# Google Drive dosya ID'leri
effb0_id = "1ZuqUMSsf2qo4Sp6_KPeu7Q-cDR8c5Au2"
yolov8m_id = "1uy24ojq7ZeUuZg6lcy2fuHGP8cC7htCR"
video_id="1w1DP4xfPfenbgOtfn7gdpBD2ZaDZ3ouD"
# Dosya indirme fonksiyonu
def download_file_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download: {url}\nError: {e}")

# Output dosya yolları
effb0_output = "effb0_best.pth"
yolov8m_output = "yolov8m.pt"
video_output="test.mp4"
# Dosyaları indir
download_file_from_google_drive(effb0_id, effb0_output)
download_file_from_google_drive(yolov8m_id, yolov8m_output)
download_file_from_google_drive(video_id, video_output)
