# --- AI画像収集ツール開発: Webアプリ版 バックエンド (Ver. 2.1) ---

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
# 例: /results/20250925_153000/000001.jpg のようなURLで画像にアクセスできるようにします
@app.route('/results/<path:subpath>')
def serve_results(subpath):
    return send_from_directory(DOWNLOAD_FOLDER, subpath)

# --- 画像収集とフィルタリング処理 ---
@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.json
    keyword = data.get('keyword')
    max_num = int(data.get('max_num', 10))
    # UIから最小・最大人数を受け取るように変更
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

        # AIフィルタリング
        valid_images = []
        image_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
        total_images = len(image_files)

        for filename in image_files:
            image_path = os.path.join(save_path, filename)
            person_count = count_persons_in_image(image_path)
            
            # UIから指定された人数でフィルタリング
            if min_persons <= person_count <= max_persons:
                valid_images.append(filename)
            else:
                os.remove(image_path) # 条件に合わない画像は削除
        
        # 有効な画像がなかった場合
        if not valid_images:
            return jsonify({
                'status': 'success',
                'message': f'{total_images}枚収集しましたが、条件（{min_persons}〜{max_persons}人）に合う画像は見つかりませんでした。',
                'imageUrls': [],
                'downloadUrl': None
            })

        # 有効な画像をZIPファイルに圧縮
        zip_filename = f"{session_id}_{keyword.replace(' ', '_')}.zip"
        zip_filepath = os.path.join(ZIP_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for image_file in valid_images:
                zipf.write(os.path.join(save_path, image_file), arcname=image_file)

        # Webページに返すURLを生成
        download_url = f'/download_zip/{zip_filename}'
        image_urls = [f'/results/{session_id}/{filename}' for filename in valid_images]
        
        # 正常な応答を返す
        return jsonify({
            'status': 'success',
            'message': f'{total_images}枚中 {len(valid_images)}枚の画像を収集しました。',
            'imageUrls': image_urls, # 画像プレビュー用のURLリストを追加
            'downloadUrl': download_url
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'エラーが発生しました: {str(e)}'}), 500

# --- ZIPファイルダウンロード処理 ---
# URLを /download_zip/ に変更
@app.route('/download_zip/<filename>')
def download_zip_file(filename):
    return send_from_directory(ZIP_FOLDER, filename, as_attachment=True)

# --- サーバー起動 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

