from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to capture face images
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'GET':
        # Serve the capture form
        return render_template('capture.html')
    elif request.method == 'POST':
        # Handle the capture request
        data = request.get_json()
        user_name = data.get('userName', None)

        if not user_name:
            return jsonify({'message': 'User name is required!'}), 400

        try:
            # Run the face_taker.py script with the user name as an argument
            subprocess.run(['python', 'face_taker.py', user_name], check=True)
            return jsonify({'message': f'Face images for {user_name} captured successfully!'})
        except subprocess.CalledProcessError as e:
            return jsonify({'message': f'Error capturing face images: {str(e)}'}), 500

# Route to train the face recognition model
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        # Serve the train page
        return render_template('train.html')
    elif request.method == 'POST':
        try:
            # Run the face_train.py script
            subprocess.run(['python', 'face_train.py'], check=True)
            return jsonify({'message': 'Model trained successfully!'})
        except subprocess.CalledProcessError as e:
            return jsonify({'message': f'Error training model: {str(e)}'}), 500

# Route to recognize faces in real time
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'GET':
        # Serve the recognize page
        return render_template('recognize.html')
    elif request.method == 'POST':
        try:
            # Run the face_recognizer.py script
            subprocess.run(['python', 'face_recognizer.py'], check=True)
            return jsonify({'message': 'Recognition started successfully!'})
        except subprocess.CalledProcessError as e:
            return jsonify({'message': f'Error starting recognition: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
