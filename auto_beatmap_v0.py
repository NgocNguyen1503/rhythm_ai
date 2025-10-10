import librosa
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import yt_dlp

# ========== 1. HÀM LÀM SẠCH TÊN FILE ==========

def sanitize_filename(filename):
    """Loại bỏ ký tự không hợp lệ trong tên file (/, :, ?, *, |, <, >, ...)"""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


# ========== 2. TẢI NHẠC TỪ YOUTUBE ==========

def download_audio_from_youtube(url, output_audio_dir="downloads/audio"):
    os.makedirs(output_audio_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_audio_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("🔽 Đang tải âm thanh từ YouTube...")
        info = ydl.extract_info(url, download=True)
        title = sanitize_filename(info['title'])
        filename = os.path.join(output_audio_dir, f"{title}.mp3")

        # Đổi tên nếu cần
        for f in os.listdir(output_audio_dir):
            if f.endswith(".mp3") and not os.path.exists(filename):
                try:
                    os.rename(os.path.join(output_audio_dir, f), filename)
                except Exception:
                    pass

    print("✅ Tải xong:", filename)
    return filename, title


# ========== 3. PHÂN TÍCH NHỊP & NĂNG LƯỢNG ==========

def extract_beats(audio_path):
    print("🎵 Đang phân tích nhạc:", audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Năng lượng (độ mạnh từng thời điểm)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Match từng beat với năng lượng gần nhất
    beat_strength = []
    for t in beat_times:
        idx = (np.abs(onset_times - t)).argmin()
        beat_strength.append(onset_env[idx])

    # Chuẩn hóa năng lượng về [0,1]
    beat_strength = np.array(beat_strength)
    beat_strength = (beat_strength - beat_strength.min()) / (beat_strength.max() - beat_strength.min() + 1e-9)

    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0])

    print(f"🎚️ Tempo trung bình: {tempo:.2f} BPM - Tổng số nhịp: {len(beat_times)}")
    return beat_times, beat_strength, tempo, y, sr


# ========== 4. SINH VỊ TRÍ CỐ ĐỊNH (KHÔNG RANDOM) ==========

def generate_positions(num_beats):
    """Sinh vị trí tuần hoàn (trái – phải – giữa – trái – phải – giữa ...)"""
    pattern = [(0.3, 0.7), (0.7, 0.7), (0.5, 0.4)]
    return [pattern[i % len(pattern)] for i in range(num_beats)]


# ========== 5. TẠO FILE BEATMAP JSON ==========

def generate_beatmap_json(beat_times, beat_strength, song_title, difficulty, beatmap_dir="downloads/beatmaps"):
    os.makedirs(beatmap_dir, exist_ok=True)
    output_path = os.path.join(beatmap_dir, f"{song_title}_{difficulty}.json")

    positions = generate_positions(len(beat_times))
    beatmap_data = {"difficulty": difficulty, "beats": []}

    for i, t in enumerate(beat_times):
        energy = round(float(beat_strength[i]), 3)
        x, y = positions[i]
        beatmap_data["beats"].append({
            "time": round(float(t), 3),
            "energy": energy,
            "x": x,
            "y": y
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(beatmap_data, f, ensure_ascii=False, indent=4)

    print(f"💾 Đã lưu beatmap ({difficulty}) tại:", output_path)
    return output_path


# ========== 6. VẼ VÀ LƯU WAVEFORM + BEAT ==========

def save_waveform_plot(y, sr, beat_times, beat_strength, tempo, song_title, output_dir="downloads/waveforms"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{song_title}_waveform.png")

    plt.figure(figsize=(12, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, alpha=0.5, color='gray', label="Waveform")
    plt.vlines(beat_times, ymin=-1, ymax=1, color='dodgerblue', alpha=0.7, linewidth=1.5, label="Beats")

    # Vẽ năng lượng
    for t, s in zip(beat_times, beat_strength):
        plt.vlines(t, ymin=-s, ymax=s, color='orange', alpha=0.8)

    plt.title(f"{song_title} — Waveform + Beat ({tempo:.2f} BPM)")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Biên độ / Năng lượng")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🖼️ Đã lưu ảnh waveform tại: {output_path}")


# ========== 7. MAIN ==========

def main():
    print("=== 🎧 AI Auto Beatmap Generator ===")
    inp = input("👉 Nhập link YouTube hoặc đường dẫn file nhạc (.mp3/.wav): ").strip()

    # Nếu là YouTube
    if "youtube.com" in inp or "youtu.be" in inp:
        audio_path, song_title = download_audio_from_youtube(inp)
    else:
        audio_path = inp
        if not os.path.exists(audio_path):
            print("❌ Không tìm thấy file nhạc!")
            return
        song_title = sanitize_filename(os.path.splitext(os.path.basename(audio_path))[0])

        os.makedirs("downloads/audio", exist_ok=True)
        new_audio_path = os.path.join("downloads/audio", f"{song_title}.mp3")
        if audio_path != new_audio_path:
            os.rename(audio_path, new_audio_path)
        audio_path = new_audio_path

    # Phân tích nhịp
    beat_times, beat_strength, tempo, y, sr = extract_beats(audio_path)

    # Tạo 3 cấp độ
    difficulties = {
        "easy": (beat_times[::3], beat_strength[::3]),
        "normal": (beat_times[::2], beat_strength[::2]),
        "hard": (beat_times, beat_strength),
    }

    for diff, (bt, bs) in difficulties.items():
        generate_beatmap_json(bt, bs, song_title, diff)

    # Lưu hình waveform
    save_waveform_plot(y, sr, beat_times, beat_strength, tempo, song_title)

    print(f"\n🎯 Hoàn tất! Beatmap cho '{song_title}' gồm {len(beat_times)} nhịp.")
    print(f"→ Nhạc: {audio_path}")
    print(f"→ Beatmaps lưu tại: downloads/beatmaps/")
    print(f"→ Waveform lưu tại: downloads/waveforms/{song_title}_waveform.png")


if __name__ == "__main__":
    main()