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
def extract_beats(audio_path, energy_threshold=0.03):
    print("🎵 Đang phân tích nhạc:", audio_path)
    y, sr = librosa.load(audio_path, sr=None)

    # onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # năng lượng RMS để loại bỏ đoạn yên
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    valid_times, valid_strength = [], []
    for t in onset_times:
        idx = np.argmin(np.abs(rms_times - t))
        e = float(rms[idx])
        if e > energy_threshold:
            valid_times.append(t)
            valid_strength.append(e)

    valid_times = np.array(valid_times)
    valid_strength = np.array(valid_strength)
    if len(valid_strength) > 0:
        valid_strength = (valid_strength - valid_strength.min()) / (valid_strength.max() - valid_strength.min() + 1e-9)

    tempo = librosa.beat.tempo(y=y, sr=sr)
    tempo = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

    print(f"🎚️ Tempo ước lượng: {tempo:.2f} BPM - Onset hợp lệ: {len(valid_times)}")
    return valid_times, valid_strength, tempo, y, sr, rms, rms_times


# ========== 4. SINH BEATMAP ==========
def generate_beatmap_json(beat_times, beat_strength, rms, rms_times, song_title, difficulty, beatmap_dir="downloads/beatmaps"):
    os.makedirs(beatmap_dir, exist_ok=True)
    output_path = os.path.join(beatmap_dir, f"{song_title}_{difficulty}.json")

    beatmap_data = {"difficulty": difficulty, "beats": []}

    # độ khó → sample density
    if difficulty == "easy":
        step, double_p, triple_p = 3, 0.05, 0.00
    elif difficulty == "normal":
        step, double_p, triple_p = 2, 0.15, 0.05
    else:
        step, double_p, triple_p = 1, 0.25, 0.10

    sample_times = beat_times[::step]
    sample_strength = beat_strength[::step] if len(beat_strength) > 0 else np.zeros_like(sample_times)

    if len(sample_times) == 0:
        print("⚠️ Không có beat hợp lệ.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(beatmap_data, f, ensure_ascii=False, indent=4)
        return output_path

    intervals = np.diff(sample_times)
    median_interval = float(np.median(intervals)) if len(intervals) > 0 else 0.5

    min_gap, min_hold, energy_hold_ratio = 0.06, 0.35, 0.6

    for i, (t, e) in enumerate(zip(sample_times, sample_strength)):
        if e < 0.05:  # bỏ đoạn yên
            continue

        r = random.random()
        if r < triple_p and e > 0.7:
            count = 3
        elif r < double_p + triple_p and e > 0.5:
            count = 2
        else:
            count = 1

        lanes = random.sample([1, 2, 3, 4], count)
        next_t = sample_times[i+1] if i < len(sample_times)-1 else None

        for lane in lanes:
            # ——— kiểm tra xem âm này có sustain không (Cách 1: năng lượng dài hạn)
            idx = np.argmin(np.abs(rms_times - t))
            window_dur = 0.5  # khoảng kiểm tra 0.5s sau onset
            end_idx = np.argmin(np.abs(rms_times - (t + window_dur)))
            energy_window = rms[idx:end_idx] if end_idx > idx else np.array([rms[idx]])
            sustain_ratio = np.mean(energy_window) / (rms[idx] + 1e-9)

            want_hold = (sustain_ratio > energy_hold_ratio)

            if want_hold:
                duration = window_dur * sustain_ratio * random.uniform(0.8, 1.5)
                if next_t:
                    allowed = max(0.0, next_t - t - min_gap)
                    duration = min(duration, allowed)
                if duration < min_hold:
                    note_type = "tap"
                    duration = 0.0
                else:
                    note_type = "hold"
            else:
                note_type = "tap"
                duration = 0.0

            note = {
                "time": round(float(t), 3),
                "lane": int(lane),
                "type": note_type,
                "energy": round(float(e), 3)
            }
            if note_type == "hold":
                note["duration"] = round(float(duration), 3)

            beatmap_data["beats"].append(note)

    # sắp xếp theo thời gian
    beatmap_data["beats"].sort(key=lambda n: (n["time"], n["lane"]))

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
        lane, t = note["lane"], note["time"]
        if note["type"] == "hold":
            dur = note.get("duration", 0.0)
            plt.plot([lane, lane], [t, t + dur], color=color, linewidth=4, alpha=0.8)
            plt.scatter(lane, t, s=20, color='black')
        else:
            plt.scatter(lane, t, s=20, color=color, alpha=0.8)

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
    print("=== 🎧 AI Auto Beatmap Generator v6 (Natural Hold Detection) ===")
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

    beat_times, beat_strength, tempo, y, sr, rms, rms_times = extract_beats(audio_path)

    difficulties = ["easy", "normal", "hard"]
    for diff in difficulties:
        path = generate_beatmap_json(beat_times, beat_strength, rms, rms_times, song_title, diff)
        with open(path, "r", encoding="utf-8") as f:
            beatmap_data = json.load(f)
        save_preview(song_title, diff, beatmap_data)

    save_waveform_plot(y, sr, beat_times, tempo, song_title)
    print("\n✅ Hoàn tất! Xem kết quả trong thư mục 'downloads/'.")


if __name__ == "__main__":
    main()