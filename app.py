import os
import uuid
import math
import flask
import urllib
from PIL import Image
import tensorflow as tf
import matplotlib as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file, flash
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model1 = load_model(
    "model.h5",compile=False)
model2 = load_model(
    "new_bimodel.h5",compile=False)



ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif', 'JPG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
classes = ['Catla', 'Mori', 'Rohu']

# @app.route('/' , methods = ['POST'])
# def estm_catch():
#     if request.method == "POST":
#         # getting input with name = fname in HTML form
#         weight = request.form.get("estmp")
#         print(weight)
#     return render_template('index.html', estm = weight)
def check(filename, model):
    # print(1)
    # img = cv2.imread(filename)
    # resize = tf.image.resize(img, (256, 256))
    # resize.numpy().astype(int)
    # yhat = model.predict(np.expand_dims(resize / 255, 0))
    # print(yhat)
    img = load_img(filename, target_size=(128, 128, 3))
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)
    print(np.argmax(result))
    yhat = np.argmax(result)
    if yhat ==0:
        return True
    else:
         return False
def predict(filename , model):
    img = load_img(filename , target_size=(128, 128, 3))
    img = img_to_array(img)
    img = img.reshape(1, 128 ,128 ,3)


    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)
    dict_result = {}
    for i in range(3):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result


@app.route('/')
def home():
        return render_template("index.html")

@app.route('/first')
def first():
        return render_template("first.html")

@app.route('/about')
def about():
        return render_template("about.html")

@app.route('/contact')
def contact():
        return render_template("contact.html")


@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''

    
    # if weight<=0:
    #     ('Weight cant be less than or equal to zero')
    #print(type(weight))
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('first.html', error=error)

        elif (request.files):
            file = request.files['file']
            print(file.filename)
            if file and allowed_file(file.filename):
                print("saving\n")
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                if check(img_path, model2):
                    class_result , prob_result = predict(img_path , model1)

                    predictions = {
                            "class1":class_result[0],
                            "class2":class_result[1],
                            "class3":class_result[2],
                            "prob1": (prob_result[0]),
                            "prob2": prob_result[1],
                            "prob3": prob_result[2]
                    }
                else:
                    error = ("The image is not recognized")
                    return render_template('first.html',error = error)
            else:

                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('first.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True, port=8000)


