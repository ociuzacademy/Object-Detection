#import libraries
import numpy as np
import time
import torch
import cv2
from fpdf import FPDF
import os
from flask import Flask, request, jsonify, render_template, Response
import pickle
import pandas as pd







#Initialize the flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

output_folder = 'Result'
os.makedirs(output_folder, exist_ok=True)

fps = 0.0
angle = 0.0 



if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Load the YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
#gun_model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
#knife_model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/gun_knife.pt')
#hammer_model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
#smoke_model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/firesmoke.pt')
#fire_model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/firesmoke.pt')






def generate_frames():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
    global fps
    camera = cv2.VideoCapture(1)
    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frame_width)

    prev_time = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time


            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform inference
            results = model(rgb_image)
            # Extract detections
            detections = results.xyxy[0].cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2, confidence, class]
            confidence_threshold = 0.45  # Set confidence threshold to 60%
            # Flag to track if any detection is made

            detection_made = False
            # Draw bounding boxes and labels
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if conf >= confidence_threshold:
                    detection_made = True
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    # Draw rectangle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put label
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If no detections are made, write "No Detection" on the frame
            if not detection_made:
                 text = "No Detection"
                 text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                 text_x = (frame.shape[1] - text_size[0]) // 2
                 text_y = (frame.shape[0] + text_size[1]) // 2
                 cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



            

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')






#default page of our web-app
@app.route('/')
def landing():
    return render_template('1.landing.html')


@app.route('/signup',methods=['POST'])
def signup():
    if request.method == 'POST':
        return render_template('2.signup.html')

@app.route('/signupsuccess',methods=['POST'])
def signupsuccess():
    if request.method == 'POST':
        credentials = [(x) for x in request.form.values()]
        print(credentials)
        username = credentials[0]
        password = credentials[1]


        file = open("Username/username.txt", "w")
        a = file.write(username)
        file.close()


        file = open("Password/password.txt", "w")
        a = file.write(password)
        file.close()

        return render_template('3.signupsuccess.html')


@app.route('/login',methods=['POST'])
def login():
    if request.method == 'POST':
        return render_template('4.login.html')



@app.route('/home',methods=['POST'])
def home():
    if request.method == 'POST':
        lcredentials = [(x) for x in request.form.values()]
        print(lcredentials)
        lusername = lcredentials[0]
        lpassword = lcredentials[1]
        #print(type(lusername))

        f = open("Username/username.txt", "r")
        username = f.read()
        f = open("Password/password.txt", "r")
        password = f.read()

        print(lusername, username, lpassword, password)

        if username==lusername and password==lpassword:
            print('match')
            template = '6.home.html'
        elif username!=lusername or password!=lpassword:
            print('No')
            template = '5.loginfailed.html'

        return render_template(template)




@app.route('/home2',methods=['POST'])
def home2():
    if request.method == 'POST':
        return render_template('6.home.html')



@app.route('/searchgun',methods=['POST'])
def searchgun():
    if request.method == 'POST':
        return render_template('7.searchgun.html')



@app.route('/searchhammer',methods=['POST'])
def searchhammer():
    if request.method == 'POST':
        return render_template('8.searchhammer.html')


@app.route('/searchsmoke',methods=['POST'])
def searchsmoke():
    if request.method == 'POST':
        return render_template('9.searchsmoke.html')


@app.route('/searchknife',methods=['POST'])
def searchknife():
    if request.method == 'POST':
        return render_template('11.searchknife.html')

@app.route('/searchfire',methods=['POST'])
def searchfire():
    if request.method == 'POST':
        return render_template('10.searchfire.html')


@app.route('/search',methods=['POST'])
def search():
    if request.method == 'POST':
        return render_template('12.search.html')


@app.route('/video_feed')
def video_feed():
    print('YESSSSSSSSSSSSSSSSSS')
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fps')
def get_fps():
    global fps
    return jsonify(fps=fps)


# Result pages

@app.route('/process_search', methods=['POST'])
def process_search():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.2
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"

                # Draw rectangle on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put label on frame
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write detection details to the PDF (one line for each detection)
                pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')






######################################################################
@app.route('/process_gun', methods=['POST'])
def process_gun():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.2
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                class_name = model.names[int(cls)]
                if class_name=='Gun':

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Write detection details to the PDF (one line for each detection)
                    pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')




##########################################################################################################
@app.route('/process_knife', methods=['POST'])
def process_knife():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/gun_knife.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.45
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0]

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                class_name = model.names[int(cls)]
                print(class_name)
                if class_name=='knife':
                    # Draw rectangle on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put label on frame
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Write detection details to the PDF (one line for each detection)
                    pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')




##########################################################################################################
@app.route('/process_fire', methods=['POST'])
def process_fire():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/firesmoke.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.2
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                class_name = model.names[int(cls)]
                if class_name=='fire':
                    # Draw rectangle on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put label on frame
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Write detection details to the PDF (one line for each detection)
                    pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')



##########################################################################################################
@app.route('/process_smoke', methods=['POST'])
def process_smoke():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/firesmoke.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.2
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                class_name = model.names[int(cls)]
                if class_name=='smoke':
                    # Draw rectangle on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put label on frame
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # Write detection details to the PDF (one line for each detection)
                    pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')




##########################################################################################################
@app.route('/process_hammer', methods=['POST'])
def process_hammer():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_models/best_multiclass.pt')
    # Get the video file from the request
    file = request.files['file']
    video_path = os.path.join(output_folder, file.filename)
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer to save the output video
    output_video_path = os.path.join(output_folder, 'output_video_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Create PDF for storing detection log
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Write the header to the PDF
    pdf.cell(200, 10, txt="Detection Log", ln=True, align='C')
    
    # Confidence threshold
    confidence_threshold = 0.2
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Convert frame from BGR to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(rgb_frame)

        # Extract detections (bounding boxes: [x1, y1, x2, y2, confidence, class])
        detections = results.xyxy[0].cpu().numpy()

        # Get current time (in seconds) for this frame
        detection_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

        # Draw bounding boxes and labels, and save detection details to PDF
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf >= confidence_threshold:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                class_name = model.names[int(cls)]
                if class_name=='Hammer':
                    # Draw rectangle on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put label on frame
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # Write detection details to the PDF (one line for each detection)
                    pdf.cell(200, 10, txt=f"Time: {detection_time} - Detected: {label}", ln=True)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Save the PDF file with the detection log
    pdf_output_path = os.path.join(output_folder, 'detection_log.pdf')
    pdf.output(pdf_output_path)

    # Release resources
    cap.release()
    out.release()

    # Respond with the paths to the output video and PDF
    return render_template('12.result.html')

if __name__ == "__main__":
    app.run(debug=True)
