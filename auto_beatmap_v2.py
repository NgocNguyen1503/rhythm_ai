import librosa
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import yt_dlp
import random

# ========== 1. LÀM SẠCH TÊN FILE ==========
def sanitize_filename(filename):
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
        for f in os.listdir(output_audio_dir):
            if f.endswith(".mp3") and not os.path.exists(filename):
                try:
                    os.rename(os.path.join(output_audio_dir, f), filename)
                except Exception:
                    pass

    print("✅ Tải xong:", filename)
    return filename, title


# ========== 3. PHÂN TÍCH BEAT (TỰ NHIÊN HƠN) ==========
def extract_beats(audio_path):
    print("🎵 Đang phân tích nhạc:", audio_path)
    y, sr = librosa.load(audio_path, sr=None)

    # Phát hiện onset (điểm âm thanh thật sự)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Năng lượng RMS để loại bỏ đoạn yên
    energy = librosa.feature.rms(y=y)[0]
    energy_times = librosa.frames_to_time(range(len(energy)), sr=sr)

    # Lọc bỏ onset quá yên
    valid_times = []
    valid_strength = []
    for t in onset_times:
        idx = np.argmin(np.abs(energy_times - t))
        e = energy[idx]
        if e > 0.03:  # bỏ đoạn im lặng hoặc fade-out
            valid_times.append(t)
            valid_strength.append(e)

    valid_times = np.array(valid_times)
    valid_strength = np.array(valid_strength)
    valid_strength = (valid_strength - valid_strength.min()) / (valid_strength.max() - valid_strength.min() + 1e-9)

    tempo = librosa.beat.tempo(y=y, sr=sr)
    tempo = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

    print(f"🎚️ Tempo ước lượng: {tempo:.2f} BPM - Số nốt thực tế: {len(valid_times)}")
    return valid_times, valid_strength, tempo, y, sr


# ========== 4. SINH BEATMAP ==========
def generate_beatmap_json(beat_times, beat_strength, song_title, difficulty, beatmap_dir="downloads/beatmaps"):
    os.makedirs(beatmap_dir, exist_ok=True)
    output_path = os.path.join(beatmap_dir, f"{song_title}_{difficulty}.json")

    beatmap_data = {"difficulty": difficulty, "beats": []}

    # Tham số độ khó
    if difficulty == "easy":
        double_prob, triple_prob, hold_prob = 0.05, 0.0, 0.15
        sample_step = 3
    elif difficulty == "normal":
        double_prob, triple_prob, hold_prob = 0.15, 0.05, 0.25
        sample_step = 2
    else:  # hard
        double_prob, triple_prob, hold_prob = 0.3, 0.1, 0.35
        sample_step = 1

    for i, (t, energy) in enumerate(zip(beat_times[::sample_step], beat_strength[::sample_step])):
        note_pack = []

        r = random.random()
        if r < triple_prob and energy > 0.7:
            num_notes = 3
        elif r < double_prob + triple_prob and energy > 0.5:
            num_notes = 2
        else:
            num_notes = 1

        lanes = random.sample([1, 2, 3, 4], num_notes)

        for lane in lanes:
            note_type = "hold" if random.random() < hold_prob and energy > 0.6 else "tap"
            note = {
                "time": round(float(t), 3),
                "lane": lane,
                "type": note_type,
                "energy": round(float(energy), 3)
            }
            if note_type == "hold":
                note["duration"] = round(random.uniform(0.6, 1.5), 3)
            note_pack.append(note)

        beatmap_data["beats"].extend(note_pack)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(beatmap_data, f, ensure_ascii=False, indent=4)

    print(f"💾 Đã lưu beatmap ({difficulty}) tại:", output_path)
    return output_path


# ========== 5. PREVIEW ==========
def save_preview(song_title, difficulty, beatmap_data):
    os.makedirs("downloads/previews", exist_ok=True)
    output_path = f"downloads/previews/{song_title}_{difficulty}_preview.png"

    plt.figure(figsize=(8, 6))
    plt.title(f"Preview Beatmap - {song_title} ({difficulty})")

    color = {"easy": "limegreen", "normal": "orange", "hard": "crimson"}[difficulty]

    for note in beatmap_data["beats"]:
        lane = note["lane"]
        t = note["time"]
        s = 40 if note["type"] == "hold" else 20
        plt.scatter(lane, t, s=s, color=color, alpha=0.8)

    for lx in [1, 2, 3, 4]:
        plt.axvline(x=lx, color='lightgray', linestyle='--', linewidth=1)
        plt.text(lx, -0.3, f"Lane {lx}", ha='center', fontsize=9, color='gray')

    plt.xlabel("Lane (1–4)")
    plt.ylabel("Thời gian (s)")
    plt.gca().invert_yaxis()
    plt.xlim(0.5, 4.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🖼️ Đã lưu preview tại: {output_path}")


# ========== 6. WAVEFORM ==========
def save_waveform_plot(y, sr, beat_times, tempo, song_title):
    os.makedirs("downloads/waveforms", exist_ok=True)
    output_path = f"downloads/waveforms/{song_title}_waveform.png"

    plt.figure(figsize=(12, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, color='gray', alpha=0.5)
    plt.vlines(beat_times, ymin=-1, ymax=1, color='dodgerblue', alpha=0.6, linewidth=1.2)
    plt.title(f"{song_title} — Waveform + Onsets ({tempo:.1f} BPM)")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Biên độ")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Đã lưu waveform tại: {output_path}")


# ========== 7. MAIN ==========
def main():
    print("=== 🎧 AI Auto Beatmap Generator v6 (Natural Rhythm Mode) ===")
    inp = input("👉 Nhập link YouTube hoặc đường dẫn file nhạc (.mp3/.wav): ").strip()

    if "youtube.com" in inp or "youtu.be" in inp:
        audio_path, song_title = download_audio_from_youtube(inp)
    else:
        audio_path = inp
        if not os.path.exists(audio_path):
            print("❌ Không tìm thấy file nhạc!")
            return
        song_title = sanitize_filename(os.path.splitext(os.path.basename(audio_path))[0])
        os.makedirs("downloads/audio", exist_ok=True)
        new_path = os.path.join("downloads/audio", f"{song_title}.mp3")
        if audio_path != new_path:
            os.rename(audio_path, new_path)
        audio_path = new_path

    beat_times, beat_strength, tempo, y, sr = extract_beats(audio_path)

    difficulties = ["easy", "normal", "hard"]
    for diff in difficulties:
        path = generate_beatmap_json(beat_times, beat_strength, song_title, diff)
        with open(path, "r", encoding="utf-8") as f:
            beatmap_data = json.load(f)
        save_preview(song_title, diff, beatmap_data)

    save_waveform_plot(y, sr, beat_times, tempo, song_title)
    print("\n✅ Hoàn tất! Xem kết quả trong thư mục 'downloads/'.")


if __name__ == "__main__":
    main()