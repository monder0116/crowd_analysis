import gdown
import hashlib
import os
import urllib.request

effb0_id = "1ZuqUMSsf2qo4Sp6_KPeu7Q-cDR8c5Au2"
yolov8m_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
video_id = "1w1DP4xfPfenbgOtfn7gdpBD2ZaDZ3ouD"


def download_file(url, output_path):
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download: {url}\nError: {e}")


def download_file_from_google_drive(file_id, output_path, expected_md5=None):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded: {output_path}")

        if expected_md5:
            if not check_md5(output_path, expected_md5):
                print(
                    f"MD5 hash verification failed for {output_path}. Re-downloading..."
                )
                os.remove(output_path)  # Dosyayı sil
                download_file_from_google_drive(file_id, output_path,
                                                expected_md5)  # Tekrar indir

    except Exception as e:
        print(f"Failed to download: {url}\nError: {e}")


# Dosya için MD5 doğrulama fonksiyonu
def check_md5(file_path, expected_md5):
    if not os.path.isfile(file_path):
        return False

    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        while chunk := f.read(8192):
            md5.update(chunk)
    print("md5:", md5.hexdigest())
    if md5.hexdigest() == expected_md5:
        return True
    else:
        return False


effb0_output = "effb0_best.pth"
yolov8m_output = "yolov8m.pt"
video_output = "test.mp4"

effb0_md5 = "02adb7544860a8d62a4cede5b789851d"
download_file_from_google_drive(effb0_id, effb0_output, effb0_md5)
download_file_from_google_drive(video_id, video_output)
download_file(yolov8m_url, yolov8m_output)
