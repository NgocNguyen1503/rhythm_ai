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


# ========== 3. PHÂN TÍCH BEAT ==========
def extract_beats(audio_path):
    print("🎵 Đang phân tích nhạc:", audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    beat_strength = []
    for t in beat_times:
        idx = (np.abs(onset_times - t)).argmin()
        beat_strength.append(onset_env[idx])

    beat_strength = np.array(beat_strength)
    beat_strength = (beat_strength - beat_strength.min()) / (beat_strength.max() - beat_strength.min() + 1e-9)

    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0])

    print(f"🎚️ Tempo trung bình: {tempo:.2f} BPM - Tổng số nhịp: {len(beat_times)}")
    return beat_times, beat_strength, tempo, y, sr


# ========== 4. TẠO DANH SÁCH LANE ==========
def generate_lane_sequence(num_beats):
    """Sinh ngẫu nhiên lane 1–4 cho mỗi beat"""
    return [(i % 4) + 1 for i in range(num_beats)]


# ========== 5. SINH BEATMAP CÓ TAP & HOLD ==========
def generate_beatmap_json(beat_times, beat_strength, song_title, difficulty, beatmap_dir="downloads/beatmaps"):
    os.makedirs(beatmap_dir, exist_ok=True)
    output_path = os.path.join(beatmap_dir, f"{song_title}_{difficulty}.json")

    lanes = generate_lane_sequence(len(beat_times))
    beatmap_data = {"difficulty": difficulty, "beats": []}

    for i, (t, energy) in enumerate(zip(beat_times, beat_strength)):
        note_type = "hold" if energy > 0.75 and random.random() < 0.3 else "tap"
        note = {
            "time": round(float(t), 3),
            "lane": lanes[i],
            "type": note_type,
            "energy": round(float(energy), 3)
        }

        # Nếu là hold thì thêm độ dài
        if note_type == "hold":
            hold_duration = round(float(random.uniform(0.3, 1.2)), 3)
            note["duration"] = hold_duration

        beatmap_data["beats"].append(note)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(beatmap_data, f, ensure_ascii=False, indent=4)

    print(f"💾 Đã lưu beatmap ({difficulty}) tại:", output_path)
    return output_path


# ========== 6. VẼ PREVIEW ==========
def save_preview(song_title, difficulty, beat_times, beat_strength):
    os.makedirs("downloads/previews", exist_ok=True)
    output_path = f"downloads/previews/{song_title}_{difficulty}_preview.png"

    plt.figure(figsize=(8, 6))
    plt.title(f"Preview Beatmap - {song_title} ({difficulty})")

    for i, (t, energy) in enumerate(zip(beat_times, beat_strength)):
        lane = (i % 4) + 1
        plt.scatter(lane, t, s=30 + energy * 40, color='orange' if difficulty == "normal" else "crimson" if difficulty == "hard" else "limegreen", alpha=0.8)

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


# ========== 7. VẼ WAVEFORM ==========
def save_waveform_plot(y, sr, beat_times, beat_strength, tempo, song_title):
    os.makedirs("downloads/waveforms", exist_ok=True)
    output_path = f"downloads/waveforms/{song_title}_waveform.png"

    plt.figure(figsize=(12, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, color='gray', alpha=0.5)
    plt.vlines(beat_times, ymin=-1, ymax=1, color='dodgerblue', alpha=0.6, linewidth=1.2)
    plt.title(f"{song_title} — Waveform + Beats ({tempo:.1f} BPM)")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Biên độ")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Đã lưu waveform tại: {output_path}")


# ========== 8. MAIN ==========
def main():
    print("=== 🎧 AI Auto Beatmap Generator (Unity 4-Lane Edition) ===")
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

    difficulties = {
        "easy": (beat_times[::3], beat_strength[::3]),
        "normal": (beat_times[::2], beat_strength[::2]),
        "hard": (beat_times, beat_strength),
    }

    for diff, (bt, bs) in difficulties.items():
        generate_beatmap_json(bt, bs, song_title, diff)
        save_preview(song_title, diff, bt, bs)

    save_waveform_plot(y, sr, beat_times, beat_strength, tempo, song_title)
    print("\n✅ Hoàn tất! Xem kết quả trong thư mục 'downloads/'.")


if __name__ == "__main__":
    main()