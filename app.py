
# environment name is myfarm

import joblib
import pickle
import os
import numpy as np


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import argparse
import io
from PIL import Image
import datetime
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
# import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
# import google.generativeai as genai
# from dotenv import load_dotenv
import torch
import cv2
# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import sklearn

import tensorflow as tf
import logging

# Set TensorFlow logging level to suppress warning messages
tf.get_logger().setLevel(logging.ERROR)

  
app = Flask(__name__)
  
  
app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user-system'
  
mysql = MySQL(app)
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('layout.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)


  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return render_template('index.html')
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)



# model = pickle.load(open('saved_models\\NBClassifier.pkl', 'rb'))

model = pickle.load(open('saved_models\\NBClassifier.pkl', 'rb'))




@app.route('/predict', methods=['POST'])    
def predict():
    if request.method == 'POST':
        N = request.form.get('N')
        P = request.form.get('P')
        K = request.form.get('K')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')

        # Assuming N, P, K, temperature, humidity, ph, and rainfall are defined elsewhere
        input_data = [float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]

        # Perform prediction using the loaded Naive Bayes model
        prediction = model.predict([input_data])[0]

        # You can use the prediction result in your template or further processing
        return render_template('prediction.html', s=prediction)


@app.route('/p')
def p():
    return render_template('prediction.html')



@app.route('/service')
def service():
    return render_template('service.html')



@app.route('/pages')
def pages():
    return render_template('pages.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/detail')
def detail():
    return render_template('detail.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

@app.route('/team')
def team():
    return render_template('team.html')


best_model_inceptionresnetv2 = load_model("saved_models\\inceptionresnetv2.h5")

# Class labels
class_labels = ['Africanized Honey Bees (Killer Bees)',
                'Aphids',
                'Armyworms',
                'Brown Marmorated Stink Bugs',
                'Cabbage Loopers',
                'Citrus Canker',
                'Colorado Potato Beetles',
                'Corn Borers',
                'Corn Earworms',
                'Fall Armyworms',
                'Fruit Flies',
                'Spider Mites',
                'Thrips',
                'Tomato Hornworms',
                'Western Corn Rootworms']

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded file
        print(request.files)
        file = request.files['file']
        
        # Save the file to a temporary location
        file.save('static/temp_image.jpg')

        # Load and preprocess the image
        img_path = 'static/temp_image.jpg'
        img = image.load_img(img_path, target_size=(256, 256,3))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = best_model_inceptionresnetv2.predict(img_array)

        # Get the top predicted class index
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        return render_template('dangerous_insects.html', prediction=predicted_class_label, img_path=img_path)

    # Handle GET request (show the form)
    return render_template('dangerous_insects.html')  # Adjust the template name as needed


#---------------------------------------------------------- This is for the fish name predictor -----------------------------------------------
# Load the scaler model
with open('saved_models\\fish_farm\\scaler_model.joblib', 'rb') as scaler_file:
    scaler_model = joblib.load(scaler_file)

# Load the DecisionTreeClassifier model
with open('saved_models\\fish_farm\\best_dt_model.joblib', 'rb') as dt_model_file:
    best_dt_model = joblib.load(dt_model_file)



@app.route('/predict_fish', methods=['POST', 'GET'])
def predict_fish():
    if request.method == 'POST':
        # Get input data from the request form
        ph = float(request.form['ph'])
        temperature = float(request.form['temperature'])
        turbidity = float(request.form['turbidity'])

        # Preprocess features (you may need to adjust this based on your original scaling)
        features = scaler_model.transform([[ph, temperature, turbidity]])

        # Make prediction
        prediction = best_dt_model.predict(features)

        # Render the template with the prediction result
        return render_template('fish_farm.html', prediction=prediction[0])
    else:
        # Render the template for the initial GET request
        return render_template('fish_farm.html')
    

#-------------------------------------------------------------------------------------
@app.route("/cattle")
def hello_world():
    return render_template('cattle_pred.html')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
#     cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
#     return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
#     #source = 0
#     cap = cv2.VideoCapture(0)
#     return render_template('index.html')

# function to get the frames from video (output video)

def get_frame():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = predict_img.imgpath
    image_path = folder_path+'/'+latest_subfolder+'/'+filename
    video = cv2.VideoCapture(image_path)

    # Initialize counters
    cow_count = 0
    sheep_count = 0
    cattle_count = 0

    while True:
        success, image = video.read()
        if not success:
            break

        # Use the detection results to count cows, sheep, and cattle
        # Modify this part based on the structure of your detection results
        # For example, if you have a list of detected objects where each object has a 'class' attribute
        # and 'class' is 'cow', 'sheep', or 'cattle', you can count them accordingly.

        # Example (modify according to your detection results):
        for detection in detected_objects:
            class_name = detection['class']
            confidence = detection['confidence']
            bbox = detection['bbox']

            # Draw bounding box on the image
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(image, f'{class_name} {confidence:.2f}', (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update counts
            if class_name == 'cow':
                cow_count += 1
            elif class_name == 'sheep':
                sheep_count += 1
            elif class_name == 'cattle':
                cattle_count += 1

        ret, jpeg = cv2.imencode('.jpg', image)

        # Send the counts along with the image
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() +
               f'\r\n\r\nCows: {cow_count}, Sheep: {sheep_count}, Cattle: {cattle_count}'.encode())

        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('cattle_pred.html')

    else:
        return "Invalid file format"

    
@app.route("/cattle", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            process = Popen(["python", "detect.py", '--source', filepath, "--weights","cattle.pt"], shell=True)
            process.wait()
            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                return display(f.filename)
                
                
            elif file_extension == 'mp4':
                return video_feed()

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()
    model = torch.hub.load('.', 'custom','cattle.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0")  # debug=True causes Restarting with stat
