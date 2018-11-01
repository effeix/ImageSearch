import argparse
import os
import os.path
import pickle

from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np


class ImageFinder():
    def __init__(self):
        self.index = {}
        self.model = MobileNet(weights="imagenet", classes=1000)
        self.search = []
        
        with open("imagenet.txt", "r") as imgn:
            self.imagenet = [
                f.strip("\n")[f.index(":")+2:f.rfind(",")].replace("'", "") for f in imgn.readlines()]
        
    def save_index(self, index):
        with open("index.pickle", "wb") as out_f:
            pickle.dump(index, out_f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_index(self):
        with open("index.pickle", "rb") as in_f:
            self.index = pickle.load(in_f)
        
    def build_index(self):
        print("Building index")
        
        for path in self.images():
            processed = self.build_input(path)
            prediction = self.model.predict(processed)
            
            self.index[path] = prediction[0]
            
        self.save_index(self.index)
    
    def id(self, term):
        try:
            return self.imagenet.index(term)
        except ValueError as ve:
            print("ERROR: term not recognized")
            exit()
    
    def probabilities(self, tid, top=1):
        probabilities_per_id = {
            k: v[tid] for k, v in self.index.items()
        }
        
        return sorted(probabilities_per_id.items(), key=lambda x: x[1], reverse=True)[:top] 
    
    def build_input(self, path):
        img = load_img(path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)
    
    def images(self):
        paths = []

        for d in os.listdir("./images/"):
            files = os.listdir("./images/"+d)
            paths.extend([f"./images/{d}/{f}" for f in files])

        return paths

    def do_search(self, term, how_many):
        self.search = self.probabilities(self.id(term), top=how_many)

    def display(self):
        for result in self.search:
            path = result[0]
            confidence = result[1]
            img = load_img(path)
            plt.title(f"{confidence*100:.2f}%")
            plt.imshow(img)
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("term", nargs=1)
    parser.add_argument("--build-index", action='store_true', default=False)
    args = parser.parse_args()
    args = parser.parse_args()

    print("Searching...")

    imgf = ImageFinder()

    if args.build_index:
        imgf.build_index()
    else:
        imgf.load_index()

    imgf.do_search(args.term[0], 5)
    imgf.display()
