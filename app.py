from flask import Flask, request, render_template, send_file, Response
import os
from process import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        input_path = r"static/uploads/input.mp4"
        file.save(input_path)
        return render_template('index.html',send_file=input_path)  

    
@app.route('/video_feed')
def video_feed():
    input_path = r'static/uploads/input.mp4'  
    return Response(generate_frames(input_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
