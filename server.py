from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
import os
import string
import random
import shutil
from glob import glob
import argparse

from scripts.net import *
from scripts.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", "-c", default="cpu", type=str)
args = parser.parse_args()

SAVE_DIR = "./images"
if os.path.isdir(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.mkdir(SAVE_DIR)

app = Flask(__name__, static_url_path="")

comp_img = CompImage(args.cuda)

def random_str(n):
    return "".join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route("/")
def index():
    images = convert_data(glob(SAVE_DIR + "/*.png"))
    return render_template("index.html", images=images)

@app.route("/images/<path:path>")
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

@app.route("/upload", methods=["POST"])
def upload():
    if request.files["image"]:
        stream = request.files["image"].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        fake, res = comp_img.compute_image(img)
        fake = ((fake * 127.5) + 127.5).view(fake.size()[-3:])
        fake = fake.to("cpu").detach().numpy().transpose(1,2,0).astype(np.uint8)

        age, gender, race, smile = res
        age    = int(age.to("cpu").detach().numpy().reshape(1) * 100)
        gender = int(gender.to("cpu").detach().numpy().argmax())
        race   = int(race.to("cpu").detach().numpy().argmax())
        smile  = int(smile.to("cpu").detach().numpy().reshape(1) * 100)

        rand_str = random_str(10)
        raw_path = os.path.join(SAVE_DIR, "raw_{}_{}_{}_{}_".format(age, gender, race, smile) + rand_str + "_.png")
        fake_path = os.path.join(SAVE_DIR, "fake_" + rand_str + "_.png")
        cv2.imwrite(raw_path, img)
        cv2.imwrite(fake_path, fake)

    return redirect("/")

if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=8000)
    
