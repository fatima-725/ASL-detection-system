import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow
import keras
class SignLanguageGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("ASL DETECTION SYSTEM")
        self.window.geometry("800x600")

        # Set a custom font for the window title

        background_image = Image.open("net.jpg")
        self.bg_image = ImageTk.PhotoImage(background_image)
        background_label = tk.Label(self.window, image=self.bg_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.prediction_label = ttk.Label(self.window, text="AMERICAN SIGN LANGUAGE DETECTION SYSTEM",
                                          font=("Times New Roman", 24, "bold"), background="black", foreground="white")
        self.prediction_label.pack(padx=10, pady=(10, 10))
        self.alphabet_var = tk.StringVar()

        self.create_widgets()

        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("ModelCN.h5", "labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["A", "B", "C", "D", "E"]

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Times New Roman", "16", "bold italic"), foreground="white" )
        style.configure("TFrame", background="#ECECEC")

        self.video_frame = ttk.Frame(self.window, style="TFrame")
        self.video_frame.pack(padx=5, pady=5)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()




        # self.alphabet_label = ttk.Label(self.window, text="", font=("Times New Roman", 24))
        # self.alphabet_label.pack(padx=10, pady=(10, 10))

        self.alphabet_value_label = ttk.Label(self.window, textvariable=self.alphabet_var, font=("Times New Roman", 24),background="black")
        self.alphabet_value_label.pack(padx=10, pady=(10, 10))

        self.alphabet_label = ttk.Label(self.window, text="PRESENTED BY:", font=("Times New Roman", 16), background="black")
        self.alphabet_label.place(x=1050, y=420)
        self.alphabet_label = ttk.Label(self.window, text="AMBER ASIF (F20CSC02)", font=("Times New Roman", 16), background="black")
        self.alphabet_label.place(x=1050, y=460)
        self.alphabet_label = ttk.Label(self.window, text="UZMA FATIMA (F20CSC03)", font=("Times New Roman", 16), background="black")
        self.alphabet_label.place(x=1050, y=500)
        self.alphabet_label = ttk.Label(self.window, text="FATIMA TUZ ZAHRA (F20CSC07)", font=("Times New Roman", 16), background="black")
        self.alphabet_label.place(x=1050, y=540)

    def update_frame(self):
        success, img = self.cap.read()
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        index = -1
        if hands:
            hands = hands[0]
            x, y, w, h = hands['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                          (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                          (255, 0, 255), 4)


        # Convert the image to PIL format for displaying in GUI
        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        imgPil = Image.fromarray(imgOutput)
        imgTk = ImageTk.PhotoImage(imgPil)

        # Update the video label with the new image
        self.video_label.configure(image=imgTk)
        self.video_label.image = imgTk
        self.alphabet_var.set(f"ALPHABET: {self.labels[index]}")
        # Schedule the next update after a delay
        self.window.after(10, self.update_frame)

    def start(self):
        self.update_frame()
        self.window.mainloop()
if __name__ == "__main__":
    gui = SignLanguageGUI()
    gui.start()
