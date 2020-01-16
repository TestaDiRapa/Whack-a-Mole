from flask import Flask, render_template, request
from whackamole import clustering, do_nothing, find_contours, jaccard_index, preprocessing
from whackamole.molenet import mole_net_predict
import cv2 as cv
import numpy as np
import os
import time

app = Flask(__name__)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():

    base_path = "static/img/result/"
    temps = os.listdir(base_path)
    if len(temps) > 0:
        os.remove(base_path + temps[0])

    out_filename = "result{}.png".format(time.time())

    op = request.form.get("algorithm")
    lesion_form = request.files["lesion"].read()
    lesion_arr = np.frombuffer(lesion_form, np.uint8)
    lesion = cv.imdecode(lesion_arr, cv.IMREAD_COLOR)
    lesion = cv.resize(lesion, (600, 400))

    mask = None
    if "mask" in request.files and request.files["mask"].filename != "" and op != "PP":
        mask_form = request.files["mask"].read()
        mask_arr = np.frombuffer(mask_form, np.uint8)
        mask = cv.imdecode(mask_arr, cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask, (600, 400))

    algorithm = do_nothing
    jaccard = -1

    if op == "COCOAW":
        algorithm = find_contours
    elif op == "IDC":
        algorithm = clustering
    elif op == "MN":
        algorithm = mole_net_predict

    start = time.time()
    processed_lesion = preprocessing(lesion)
    predicted_mask = algorithm(processed_lesion)
    end = time.time()

    if mask is not None:
        contours_real, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(lesion, contours_real, -1, (0, 0, 255), 2, 4)
        jaccard = round(jaccard_index(predicted_mask, mask), 3)

    if op != "PP":
        contours_pre, _ = cv.findContours(predicted_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(lesion, contours_pre, -1, (255, 0, 0), 2, 4)
        cv.imwrite(base_path + out_filename, lesion)
    else:
        cv.imwrite(base_path + out_filename, processed_lesion)

    return render_template("result.html",
                           image_filename=out_filename,
                           processing_time=int(end - start),
                           jaccard=jaccard,
                           prediction=(op != "PP"),
                           masked=(mask is not None))


if __name__ == '__main__':
    app.run('127.0.0.1', port=8080)
