# --- AI画像収集ツール開発: Webアプリ版 バックエンド (Ver. 2.2 - バッチ処理対応) ---

# 必要なライブラリを読み込みます
from flask import Flask, request, jsonify, render_template, send_from_directory
from icrawler.builtin import GoogleImageCrawler
import os
import shutil
import zipfile
from datetime import datetime
from ultralytics import YOLO
import torch

# Flaskアプリケーションを作成します
app = Flask(__name__, static_folder='static', template_folder='.')

# 画像を一時的に保存するフォルダ
DOWNLOAD_FOLDER = 'downloads'
# ZIPファイルを保存するフォルダ
ZIP_FOLDER = 'zips'

# 必要なフォルダが存在しない場合は作成します
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)
if not os.path.exists(ZIP_FOLDER):
    os.makedirs(ZIP_FOLDER)

# --- AIモデルの読み込み ---
print("AIモデルを読み込んでいます...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用するデバイス: {device}")
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("AIモデルの読み込みが完了しました。")
except Exception as e:
    print(f"AIモデルの読み込み中にエラーが発生しました: {e}")
    model = None

def count_persons_in_image(image_path):
    if model is None: return 0
    try:
        results = model(image_path, verbose=False)
        person_count = 0
        for result in results:
            person_indices = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
            person_count += len(person_indices)
        return person_count
    except Exception as e:
        print(f"画像解析中にエラー: {image_path}, Error: {e}")
        return 0

# --- ルートURL ('/') ---
@app.route('/')
def index():
    return render_template('image_crawler_ui.html')

# --- 新機能: 収集した画像を表示するためのルート ---
@app.route('/results/<path:subpath>')
def serve_results(subpath):
    return send_from_directory(DOWNLOAD_FOLDER, subpath)

# --- 画像収集とフィルタリング処理 ---
@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.json
    keyword = data.get('keyword')
    max_num = int(data.get('max_num', 10))
    min_persons = int(data.get('min_persons', 3))
    max_persons = int(data.get('max_persons', 15))

    if not keyword:
        return jsonify({'status': 'error', 'message': 'キーワードが入力されていません。'}), 400

    try:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(DOWNLOAD_FOLDER, session_id)
        if os.path.exists(save_path): shutil.rmtree(save_path)
        os.makedirs(save_path)

        # 画像収集
        google_crawler = GoogleImageCrawler(storage={'root_dir': save_path})
        google_crawler.crawl(keyword=keyword, max_num=max_num)

        # --- ここからAIフィルタリング処理をバッチ処理に最適化 ---
        valid_images = []
        image_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
        image_paths = [os.path.join(save_path, f) for f in image_files]
        total_images = len(image_files)

        if total_images > 0 and model is not None:
            print(f"{total_images}枚の画像のバッチ処理を開始します...")
            try:
                # AIモデルで全画像を一度に解析（バッチ処理）
                results = model(image_paths, verbose=False)
                print("バッチ処理が完了しました。")

                # 解析結果を元に画像をフィルタリング
                for image_file, result in zip(image_files, results):
                    person_indices = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
                    person_count = len(person_indices)
                    
                    if min_persons <= person_count <= max_persons:
                        valid_images.append(image_file)
                    else:
                        os.remove(os.path.join(save_path, image_file))
            except Exception as e:
                print(f"AIのバッチ処理中にエラーが発生しました: {e}")
                return jsonify({'status': 'error', 'message': f'AIの画像解析中にエラーが発生しました: {str(e)}'}), 500
        elif model is None:
            print("AIモデルが利用できないため、フィルタリングをスキップします。")
            valid_images = image_files
        # --- 最適化ここまで ---
        
        if not valid_images:
            return jsonify({
                'status': 'success',
                'message': f'{total_images}枚収集しましたが、条件（{min_persons}〜{max_persons}人）に合う画像は見つかりませんでした。',
                'imageUrls': [],
                'downloadUrl': None
            })

        zip_filename = f"{session_id}_{keyword.replace(' ', '_')}.zip"
        zip_filepath = os.path.join(ZIP_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for image_file in valid_images:
                zipf.write(os.path.join(save_path, image_file), arcname=image_file)

        download_url = f'/download_zip/{zip_filename}'
        image_urls = [f'/results/{session_id}/{filename}' for filename in valid_images]
        
        return jsonify({
            'status': 'success',
            'message': f'{total_images}枚中 {len(valid_images)}枚の画像を収集しました。',
            'imageUrls': image_urls,
            'downloadUrl': download_url
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'エラーが発生しました: {str(e)}'}), 500

# --- ZIPファイルダウンロード処理 ---
@app.route('/download_zip/<filename>')
def download_zip_file(filename):
    return send_from_directory(ZIP_FOLDER, filename, as_attachment=True)

# --- サーバー起動 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

