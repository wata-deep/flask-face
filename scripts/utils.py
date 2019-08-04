import torch
import cv2
import numpy as np

from net import *

conv_gender = {"0": "male", "1": "female"}
conv_race = {"0": "White", "1": "Black", "2": "Asian", "3": "Indian"}

class CompImage:
    def __init__(self, device="cpu"):
        self.image_size = (128, 128)
        self.device = device

        h_vec_size = 128
        c_vec_size = 8

        vec_size = h_vec_size + c_vec_size

        self.gen = Gen(h_vec_size, c_vec_size).to(device)
        self.enc = Enc(h_vec_size).to(device)
        self.cnn = Net_ele().to(device)

        self.gen.load_state_dict(torch.load("models/gen.pt", map_location=device))
        self.enc.load_state_dict(torch.load("models/enc.pt", map_location=device))
        self.cnn.load_state_dict(torch.load("models/cnn.pt", map_location=device))

        self.gen.eval()
        self.enc.eval()
        self.cnn.eval()

    def create_label_vec(self, age, gender, race, smile):
        age = [int(age) / 100.]
        gender = [1. if i == int(gender) else 0. for i in range(2)]
        race = [1. if i == int(race) else 0. for i in range(4)]
        smile = [int(smile)/ 100.]

        return age + gender + race + smile

    def compute_image(self, image):
        image = cv2.resize(image, self.image_size)
        image = torch.tensor(image.transpose(2,0,1)).to(self.device)
        image = image.view(1,3,128,128).float()
    
        age = np.random.randint(1, 101)
        gender = np.random.randint(2)
        race = np.random.randint(4)
        smile = np.random.randint(1, 101)
        label = torch.tensor([self.create_label_vec(age, gender, race, smile)]).to(self.device)
        
        noise = self.enc(image)
        fake = self.gen(noise, label)
        res_age, res_gender, res_race, res_smile, _ = self.cnn(image)
    
        return fake, (res_age, res_gender, res_race, res_smile)

def convert_data(datas):
    res = {}
    datas.sort()
    for data in datas:
        data = data.split("/")[-1]
        image_type = data.split("_")[0]
        code = data.split("_")[-2]
        res.setdefault(code, {})
        if image_type == "raw":
            res[code]["raw"] = data
            res[code]["age"] = data.split("_")[1]
            res[code]["gender"] = conv_gender[data.split("_")[2]]
            res[code]["race"] = conv_race[data.split("_")[3]]
            res[code]["smile"] = data.split("_")[4]
        else:
            res[code]["fake"] = data

    return list(res.values())
