# --- AI画像収集ツール開発: Webアプリ版 バックエンド (Ver. 2.0) ---

# 必要なライブラリを読み込みます
from flask import Flask, request, jsonify, render_template, send_from_directory
from icrawler.builtin import GoogleImageCrawler
import os
import shutil
import zipfile
from datetime import datetime
from ultralytics import YOLO # --- AI機能の追加 ---
import torch # --- AI機能の追加 ---

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

# --- ここからAI機能の追加 ---

# AIモデルを起動時に一度だけ読み込みます
# 'yolov8n.pt'は、物体検出のための学習済みモデルファイルです。
# 最初の実行時に自動でダウンロードされます。
print("AIモデルを読み込んでいます...")
try:
    # デバイスの自動選択（GPUが利用可能ならGPUを、そうでなければCPUを使用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用するデバイス: {device}")
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("AIモデルの読み込みが完了しました。")
except Exception as e:
    print(f"AIモデルの読み込み中にエラーが発生しました: {e}")
    model = None

def count_persons_in_image(image_path):
    """
    指定された画像ファイル内の人物の数をカウントします。
    Args:
        image_path (str): 画像ファイルのパス
    Returns:
        int: 検出された人物の数
    """
    if model is None:
        print("AIモデルが利用できません。")
        return 0
        
    try:
        # AIモデルで画像を解析（推論）します
        results = model(image_path, verbose=False)
        
        person_count = 0
        # 解析結果を一つずつチェックします
        for result in results:
            # 検出された物体のクラスIDを取得します
            # YOLOv8の学習済みモデルでは、'person'（人物）のIDは '0' です
            person_indices = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
            person_count += len(person_indices)
            
        return person_count
    except Exception as e:
        # Pillowライブラリが対応していない画像形式などの場合
        print(f"画像解析中にエラー: {image_path}, Error: {e}")
        return 0

# --- ここまでAI機能の追加 ---


# --- ルートURL ('/') にアクセスがあった場合の処理 ---
@app.route('/')
def index():
    # image_crawler_ui.html をブラウザに表示します
    return render_template('image_crawler_ui.html')

# --- '/crawl' というURLにPOSTリクエストがあった場合の処理 ---
@app.route('/crawl', methods=['POST'])
def crawl():
    # Webページから送られてきたデータを取得します
    data = request.json
    keyword = data.get('keyword')
    max_num = int(data.get('max_num', 10))

    if not keyword:
        return jsonify({'status': 'error', 'message': 'キーワードが入力されていません。'}), 400

    try:
        # フォルダ名をセッションごとにユニークにする
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(DOWNLOAD_FOLDER, session_id)
        
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

        # 画像収集（クロール）の準備
        google_crawler = GoogleImageCrawler(storage={'root_dir': save_path})
        
        # クロール実行
        google_crawler.crawl(keyword=keyword, max_num=max_num)
        
        # ZIPファイルを作成
        zip_filename = f"{session_id}_{keyword.replace(' ', '_')}.zip"
        zip_filepath = os.path.join(ZIP_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for root, _, files in os.walk(save_path):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # ダウンロード用のURLを生成
        download_url = f'/download/{zip_filename}'
        
        return jsonify({
            'status': 'success',
            'message': f'画像の収集が完了しました。下のリンクからダウンロードしてください。',
            'downloadUrl': download_url
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'エラーが発生しました: {str(e)}'}), 500

# --- '/download/<filename>' というURLにアクセスがあった場合の処理 ---
@app.route('/download/<filename>')
def download_zip(filename):
    return send_from_directory(ZIP_FOLDER, filename, as_attachment=True)


# このスクリプトが直接実行された場合にサーバーを起動します
if __name__ == '__main__':
    app.run(debug=True, port=5001)

