from flask import Flask, request, jsonify
import os
import yt_dlp
from beatmap_generator import generate_from_input

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        name = request.json.get('name')
        audio_link = request.json.get('audio')
        input_type = request.json.get('input')

        if not audio_link or not name:
            return jsonify({
                "status": "error",
                "message": "Thiếu tham số 'name' hoặc 'audio'!"
            }), 400

        # Create folder path in Laravel server
        # Create folder for yt-dlp to save directly
        from beatmap_generator import LARAVEL_SONGS_PATH
        output_dir = os.path.join(LARAVEL_SONGS_PATH, name)
        os.makedirs(output_dir, exist_ok=True)

        # base path
        output_audio_base = os.path.join(output_dir, name)
        outtmpl = output_audio_base + ".%(ext)s"

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        print(f"Đang tải {audio_link} ...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([audio_link])

        output_audio = os.path.join(output_dir, f"{name}.mp3")
        if not os.path.exists(output_audio):
            # fallback: pick first mp3 in dir
            downloaded_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".mp3")]
            if downloaded_files:
                output_audio = os.path.join(output_dir, downloaded_files[0])
            else:
                raise FileNotFoundError("Không tìm thấy file mp3 sau khi tải!")

        print(f"Đã lưu MP3 tại {output_audio}")
        print("Bắt đầu sinh beatmap...")

        result = generate_from_input(output_audio, song_title=name)

        print("Hoàn tất sinh beatmap!")
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("- AI Beatmap Flask API (Natural Rhythm Version) -")
    print(f"Flask working dir: {os.getcwd()}")
    app.run(debug=True)