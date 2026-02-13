from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# لود کردن محیط
load_dotenv()

app = Flask(__name__)
CORS(app)  # اجازه دسترسی از هر جایی

# ===== کلید API تو =====
API_KEY = "AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0"

# تنظیم Gemini
genai.configure(api_key=API_KEY)

# انتخاب مدل
model = genai.GenerativeModel('gemini-1.5-flash')

# تاریخچه مکالمات (در حافظه - برای سادگی)
chat_histories = {}

@app.route('/')
def home():
    """صفحه اصلی چت رو نشون بده"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """دریافت پیام کاربر و برگردوندن جواب از Gemini"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', request.remote_addr)  # IP به عنوان session
        
        if not user_message:
            return jsonify({'error': 'پیام خالی است'}), 400
        
        # گرفتن تاریخچه جلسه
        history = chat_histories.get(session_id, [])
        
        # شروع چت با تاریخچه
        chat = model.start_chat(history=history)
        
        # ارسال پیام و دریافت پاسخ
        response = chat.send_message(user_message)
        
        # ذخیره تاریخچه جدید
        chat_histories[session_id] = chat.history
        
        return jsonify({
            'reply': response.text,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """پاک کردن تاریخچه یک session"""
    try:
        session_id = request.json.get('session_id', request.remote_addr)
        if session_id in chat_histories:
            del chat_histories[session_id]
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """چک کردن سلامت سرور"""
    return jsonify({
        'status': 'online',
        'model': 'gemini-1.5-flash',
        'api_key': 'valid' if API_KEY else 'missing'
    })

if __name__ == '__main__':
    print("🚀 سرور چت در حال اجراست...")
    print("📱 آدرس: http://127.0.0.1:5000")
    print("🔑 کلید API: فعال" if API_KEY else "❌ کلید API: پیدا نشد!")
    app.run(debug=True, host='0.0.0.0', port=5000)
