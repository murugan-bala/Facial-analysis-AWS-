import cv2
import boto3
import csv
import face_recognition
import glob
import os
import logging
import time
import re
import json
import sys
import flask 
import datetime
import requests
import base64
import numpy as np
import face_recognition
from flask import Flask, send_file
from flask import request
from flask import jsonify
from waitress import serve
from flask_cors import CORS
from PIL import Image, ImageDraw
#from ocrr import tesseract
#from text_tesseract import tess
app = Flask(__name__)
CORS(app)
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict

with open('rekognition_key.csv','r') as input:
    next(input)
    reader=csv.reader(input)
    for line in reader:
        access_key_id=line[2]
        secret_key_id=line[3]


client = boto3.client('rekognition',
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_key_id,
                      region_name='us-east-1'
                      )

def face_distance_to_conf(face_distance, face_match_threshold=0.60):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

#Note: OpenCV reads images in BGR format, face_recognition in RGB format, so sometimes you need to convert them, sometimes no



def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location
    if name is None:
        #count_starts = time.time()
        #name = 'Unknown'
        name = 'Face'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        #count_end = time.time()
        color = (0, 128, 0)  # dark green for recognized face
    #timee=count_starts-count_end
    #print(timee)
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

global i
i=0
def detectFaces():
    photo='image/small.jpg'   # three_peoples   muru  joker  girl properimage1 kids
    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.detect_faces(Image={'Bytes': source_bytes},Attributes=['ALL'])
    #print(response['FaceDetails'])
    #print()
    
    print("Number of faces in a given images is : ",len(response['FaceDetails']))
    return len(response['FaceDetails'])

@app.route('/face_validation',methods = ['POST','GET'])
def face_validation():
    #print("verifyPhoto.face_detection: {}".format("Entered service method"),flush=True)
    #print("Hello")
    result="No faces detected"
    output="sucess"
    if request.method == 'POST':
        try:
         #result="No faces detected"
             #output="invalid"
             error_status="Sucess"
             #print("Hellooooo face_validation")
             result="No faces detected"
             #file = request.files['file']
             req_json=request.json
             #print("face_detection.req_json: {}".format(req_json),flush=True)
             #print(type(req_json))
             if 'img1' in request.json:   #{"compare":"sucess"}
                req_json=request.json
                #print("inside validate..")
                #print("face_detection.req_json: {}".format(req_json),flush=True)
                #print(type(req_json))
                #print(req_json['img1'])
                img_name=req_json['filename']
                s=req_json['img1']
                #print("#########################################################")
                #print(s)
                x = s.split(",")
                #print(len(x))
                #print(x[1])
                encod_img1=x[1].encode()
                #encod_img1=req_json['img1'][18:].encode()
                #print(req_json['img1'][18:])  #iV
                image1_decode = base64.decodestring(encod_img1)
                with open('photo_validation.jpg', 'wb') as image1_result:
                    image1_result.write(image1_decode)
                with open('photo_validation.jpg','rb') as source_image:
                    source_bytes=source_image.read()

                response=client.detect_faces(Image={'Bytes': source_bytes},Attributes=['ALL'])
                #print(response['FaceDetails'])
                #print()
                #print(response['FaceDetails'][0]['Gender'])
                #confidence=response['FaceDetails'][0]['Gender']['Confidence']
                '''if confidence>80:
                    print("Valid Faces...")'''
                
                print("Number of faces in a given images is : ",len(response['FaceDetails']))
                face_count=len(response['FaceDetails'])
                if face_count==0:
                    result="   Invalid Face / No face detected.."
                elif face_count==1:
                    confidence=response['FaceDetails'][0]['Gender']['Confidence']
                    if confidence>80:
                        result="   Valid Face / face detected.."
                        print(response['FaceDetails'][0]['Gender'])
                    else:
                        print(response['FaceDetails'][0]['Gender'])
                        print("Face confidence not met...")
                        result="Face confidence not met..."
                elif face_count>1:
                    result="Multi face detected.."
                    #print(response['FaceDetails'][0]['Gender'])
                    #print(response['FaceDetails'][1]['Gender'])
                else:
                    result="Invalid Face"
                '''unknown_image = face_recognition.load_image_file('photo_validation.jpg')
                face_locations = face_recognition.face_locations(unknown_image) #face_locations = face_recognition.face_locations(image, model="cnn")
                print("There are ",len(face_locations),"people in this image")
                #result="single faces detected"
                img = cv2.imread('photo_validation.jpg', cv2.IMREAD_GRAYSCALE)   # my_photo, incorrect_3
                n_white_pix = np.sum(img == 255)
                print("Gray image shape is :",img.shape)
                print("Total pixels is (gray image ) :",img.size)
                print('Number of white pixels:', n_white_pix)

                if n_white_pix >50000:
                    print("Photo is not valid..")
                    result="Scaned faces detected"
                    output="invalid"
                    error_code=0
                else:
                    print("Valid photo..")
                    if len(face_locations)==0:
                        print("No Faces Detected..")
                        result="No faces detected"
                        output="invalid"
                        error_code=0
                    elif len(face_locations)>1:
                        print("More than One Face detected....")
                        result="More faces detected"
                        output="invalid"
                        error_code=2
                    elif len(face_locations)==1:
                        print("Single Face detected....")
                        result="single faces detected"
                        output="valid"
                        error_code=1'''
        except Exception as e:
            print(e)
            output="invalid"
            '''error_status=e
            result="Error Occured..."
            output="invalid"
            print("Error occured ....")
            error_code=0'''
            
    os.remove('photo_validation.jpg')        
    #return jsonify({'output' : output,'result':result,'error_code':error_code,'error_status':error_status})
    return jsonify({'output' : output,'result':result})

@app.route('/sign_validation',methods = ['POST','GET'])
def sign_validation():
    #print("verifyPhoto.face_detection: {}".format("Entered service method"),flush=True)
    #print("Hello")
    result="In-Valid Signature"
    ocr="None"
    words="None"
    output=""
    if request.method == 'POST':
        try:
         #result="No faces detected"
             #output="invalid"
             error_status="Sucess"
             #print("Hellooooo sign_validation")
             result="No faces detected"
             #file = request.files['file']
             req_json=request.json
             #print("face_detection.req_json: {}".format(req_json),flush=True)
             #print(type(req_json))
             if 'img1' in request.json:   #{"compare":"sucess"}
                req_json=request.json
                #print("inside validate..")
                #print("face_detection.req_json: {}".format(req_json),flush=True)
                #print(type(req_json))
                #print(req_json['img1'])
                #img_name=req_json['filename']
                s=req_json['img1']
                x = s.split(",")
                #print(len(x))
                #print(x[1])
                encod_img1=x[1].encode()
                #encod_img1=req_json['img1'][18:].encode()
                #print(req_json['img1'][18:])  #iV
                image1_decode = base64.decodestring(encod_img1)
                with open('sign_val.jpg', 'wb') as image1_result:
                    image1_result.write(image1_decode)
                with open('sign_val.jpg','rb') as source_image:
                    source_bytes=source_image.read()

                response=client.detect_faces(Image={'Bytes': source_bytes},Attributes=['ALL'])
                #print(response['FaceDetails'])
                #print()
                
                #print("Number of faces in a given images is : ",len(response['FaceDetails']))
                sign_count=len(response['FaceDetails'])
                if sign_count==0:
                    result="  valid Signature.."
                    print("valid Signature..")
                elif sign_count>0:
                    result="  Invalid Signature"
                    print("Invalid Signature..")
                '''unknown_image = face_recognition.load_image_file('sign_val.jpg')
                face_locations = face_recognition.face_locations(unknown_image) #face_locations = face_recognition.face_locations(image, model="cnn")
                
                img = cv2.imread('sign_val.jpg', cv2.IMREAD_GRAYSCALE)   # my_photo, incorrect_3
                n_white_pix = np.sum(img == 255)
                #print("Gray image shape is :",img.shape)
                #print("Total pixels is (gray image ) :",img.size)
                print('Number of white pixels:', n_white_pix)
                print("There are ",len(face_locations),"people in this image")
                #result="single faces detected"
                error_code=1
                if n_white_pix >50000:
                    #now = time.strftime('%d-%m-%Y %H:%M:%S')
                    print("signature is not valid..")
                    error_code=0
                    result="Invalid"
                    #printt('In_Valid signature detected : {}'.format(file))
                    #printt('{}  : signature not detected : {}'.format(now,file))
                    #in_valid=in_valid+1
                    #csvwriter.writerows([[now, file_name[-1]]])
                else:
                    #print("Valid photo..")
                    if len(face_locations)==0:
                        print("Valid Signature..")
                        result="Valid"
                        error_code=1
                    elif len(face_locations)>0:
                        print("Invalid signature........")
                        result="Invalid"
                        error_code=0'''
                '''img = cv2.imread('validation.jpg', cv2.IMREAD_GRAYSCALE)   # my_photo, incorrect_3
                n_white_pix = np.sum(img == 255)
                print("Gray image shape is :",img.shape)
                print("Total pixels is (gray image ) :",img.size)
                print('Number of white pixels:', n_white_pix)

                if n_white_pix >50000:
                    print("Photo is not valid..")
                    output="invalid"
                else:
                    print("Valid photo..")
                    output="valid"'''
        except Exception as e:
            output="invalid"
            '''error_status=e
            result="error occured.."
            print("Error occured ....")
            error_code=0'''
            
    os.remove('sign_val.jpg')        
    #return jsonify({'result':result,'error_code':error_code,'error_status':error_status})
    return jsonify({'output' : output,'result':result})
 

 

if __name__ == '__main__':
    #context = ('/etc/ssl/certs/my.crt', '/etc/ssl/private/my.key')#certificate and key files
    #app.run(host='0.0.0.0',port=5022 ,debug=True,ssl_context=context,threaded=True)
    #serve(app,host='0.0.0.0',port=5022,threads= 4)
    app.run(host='0.0.0.0',port=5030 ,debug=True,threaded=True)

