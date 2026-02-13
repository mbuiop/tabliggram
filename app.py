from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# ===== کلید API تو =====
API_KEY = "AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0"

# تنظیم Gemini
genai.configure(api_key=API_KEY)

# ===== استفاده از gemini-pro به جای flash =====
model = genai.GenerativeModel('gemini-pro')

# تاریخچه مکالمات
chat_histories = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'پیام خالی است'}), 400
        
        # گرفتن تاریخچه
        history = chat_histories.get(session_id, [])
        chat = model.start_chat(history=history)
        
        # ارسال پیام
        response = chat.send_message(user_message)
        
        # ذخیره تاریخچه
        chat_histories[session_id] = chat.history
        
        return jsonify({'reply': response.text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """لیست مدل‌های موجود رو نشون بده"""
    try:
        models = genai.list_models()
        model_list = [{'name': m.name, 'methods': m.supported_generation_methods} for m in models]
        return jsonify({'models': model_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 سرور چت در حال اجراست...")
    print("📱 آدرس: http://127.0.0.1:5000")
    print("🔑 کلید API: فعال")
    print("🤖 مدل: gemini-pro")
    app.run(debug=True, host='0.0.0.0', port=5000)
