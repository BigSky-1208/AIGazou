# --- AI画像収集ツール開発: Webアプリ版 バックエンド ---

# 必要なライブラリを読み込みます
from flask import Flask, request, jsonify, render_template, send_from_directory
from icrawler.builtin import GoogleImageCrawler
import os
import shutil
import zipfile
from datetime import datetime

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
        
        # フォルダの中身を一度空にする（古いものが残らないように）
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
