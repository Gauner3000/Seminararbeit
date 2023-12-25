import cv2
import numpy as np


def Plant_Segmentation(path):
    # Lade das Satellitenbild
    satellite_image = cv2.imread(path + "/satellite_image.png")  # Passe den Dateipfad an
    # Konvertiere das Bild in den HSV-Farbraum
    hsv_image = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2HSV)

    # Definiere den Farbbereich für Grün in HSV
    lower_green = np.array([35, 50, 30])  # Untere Schwelle für Grüntöne im HSV-Farbraum
    upper_green = np.array([90, 255, 120])  # Obere Schwelle für Grüntöne im HSV-Farbraum

    # Erzeuge eine Maske für grüne/grünliche Pixel
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Erzeuge ein schwarz-weißes Bild, auf dem Pflanzen als weiße Pixel angezeigt werden
    plant_image = np.zeros_like(satellite_image)
    plant_image[green_mask != 0] = [255, 255, 255]  # Setze alle nicht-schwarzen Pixel auf Weiß

    plant_image = cv2.resize(plant_image, (220, 220))
    # Speichere das verkleinerte schwarz-weiße Pflanzenbild
    cv2.imwrite(path + '/plant_segmented_image.png', plant_image)