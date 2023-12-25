import tkinter as tk
from tkinter import filedialog, ttk, PhotoImage
import os
import shutil
import random
import numpy as np
from PIL import Image, ImageTk
import cv2

import modelNN
import plant_calculator



def browse_file():
    global path
    file_path = filedialog.askopenfilename()
    if file_path:

        # Ordner Path
        dir_path = r'Data'
        count = 0
        # Durch Verzeichhnis Itterieren
        for path in os.listdir(dir_path):
            count += 1
        
        #Eigenen Ordner erstellen für Bild-Analyse
        path = "Data/" + str(count+1) #Path
        # Erstellen des Ordners 
        os.makedirs(path)
        # Den vollen Pfad zur Zieldatei erstellen
        ziel_pfad = path + "/satellite_image.png"
        # Das Bild kopieren und umbenennen
        shutil.copy(file_path, ziel_pfad)
        

        #Zweites Bild erstellen für spätere Darstellung
        # Öffne das Bild
        img = Image.open(ziel_pfad)
        # Ändere die Größe auf 200x200
        img.thumbnail((220, 220))
        # Speichere das komprimierte Bild
        img.save(path + "/komp_satellite_image.png")
        # Schließe das Bild
        img.close()
        # Aktuelles Fenster schließen
        first.destroy()



def update_progress_bar():
    global progress
    global counter

    if progress < 45:
        progress += random.uniform(5, 10)  # Unregelmäßiger Fortschritt

    if counter == 0:
        modelNN.Neuronal_Network(path, "street")
        modelNN.Neuronal_Network(path, "forest")
        plant_calculator.Plant_Segmentation(path)
        counter += 1

    if progress < 85:
        progress += random.uniform(2, 4)

    elif progress > 85:
        progress += random.uniform(8, 15)  # Schneller Fortschritt in den letzten 15%

    if progress >= 110:
        progress = 110

    progress_var.set(f"Fortschritt: {int(progress)}%")
    progress_bar["value"] = progress

    if progress < 110:
        second.after(110, update_progress_bar)
    else:
        # Wenn der Fortschritt 100% erreicht hat, schließe das Fenster
        second.destroy()






def percent_white_black(image_kind):
    # Finden des Pfads für die geforderten Prozentwerte
    if image_kind == "plants":
        mask = cv2.imread(path + "/plant_segmented_image.png", cv2.IMREAD_GRAYSCALE)
    elif image_kind == "forest":
        mask = cv2.imread(path + "/segmented_forest_image.png", cv2.IMREAD_GRAYSCALE)
    elif image_kind == "street":
        mask = cv2.imread(path + "/segmented_street_image.png", cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return "Fehler: Maske konnte nicht eingelesen werden."

    # Zähle die weißen Pixel in der Maske
    white_pixel_count = np.count_nonzero(mask == 255)  # Weiß ist in OpenCV 255

    # Berechne den Gesamtpixelanzahl im Bild
    total_pixel_count = mask.shape[0] * mask.shape[1]

    # Berechne den Anteil der weißen Pixel
    white_percentage = str(round((white_pixel_count / total_pixel_count) * 100, 2))
    
    return white_percentage




        

#Erstes Fenster
first = tk.Tk()
first.geometry("700x700")
first.title("Datei auswählen")
first.configure(background='white')

label = tk.Label(first, fg="black", font=('Arial', 20), text="Klicken Sie auf 'Datei auswählen', um ein Satelliten-Bild analysieren zu lassen.")
label.pack(pady=40)
label.configure(background='white')

button = tk.Button(first, fg="black", text="Datei auswählen", font=('Arial', 20), height= 5, width=18, command=browse_file)
button.pack()

#Abstand erstellen
Abstand1 = tk.Label(first, background="white", padx=10, pady=20)
Abstand1.pack()
# Lade das Bild und konvertiere es in ein unterstütztes Format (GIF)
original_image = Image.open("earth-satelite.jpg")
resized_image = original_image.resize((300, 300))
# Erstelle ein ImageTk-Objekt, um das Bild in Tkinter anzuzeigen
image = ImageTk.PhotoImage(resized_image)
# Erstelle ein Label-Widget, um das Bild anzuzeigen
label = tk.Label(first, image=image, highlightthickness=0, bg="white")
label.pack()

first.mainloop()



#Zweites Fenster
second = tk.Tk()
second.geometry("700x700")
second.title("Auswertung wird geladen")
second.configure(background='white')
progress_var = tk.StringVar()
progress_var.set("Fortschritt: 0%")


progress_bar = ttk.Progressbar(second, variable=progress_var, maximum=100, mode='determinate',
                               style="Custom.Horizontal.TProgressbar", length=680)
progress_bar.pack(pady=60, side="bottom")

# Definiere einen benutzerdefinierten ttk.Style, um Farben anzupassen
style = ttk.Style()
style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
progress = 0
counter = 0


#Abstand erstellen
Abstand1 = tk.Label(second, background="white", padx=10, pady=50)
Abstand1.pack()
#Bild darstellen
original_image = Image.open("Erde.png")
resized_image = original_image.resize((350, 350))
# Erstelle ein ImageTk-Objekt, um das Bild in Tkinter anzuzeigen
image = ImageTk.PhotoImage(resized_image)
# Erstelle ein Label-Widget, um das Bild anzuzeigen
label = tk.Label(second, image=image, highlightthickness=0, bg="white")
label.pack()

update_progress_bar()
second.mainloop()



# Drittes Fenster
third = tk.Tk()
third.title("Ergebnisse")
third.geometry("700x700")
third.configure(background='white')

# Kacheln erstellen
frame1 = tk.Frame(third, width=500, height=400, background='white')
frame1.grid(row=0, column=0)
frame2 = tk.Frame(third, width=500, height=400, background='white')
frame2.grid(row=1, column=0)
frame3 = tk.Frame(third, width=500, height=400, background='white')
frame3.grid(row=1, column=1)
frame4 = tk.Frame(third, width=500, height=400, background='white')
frame4.grid(row=0, column=1)

# Bilder laden (220x220 Pixel)
image1 = PhotoImage(file=path + "/komp_satellite_image.png")
image2 = PhotoImage(file=path + "/segmented_street_image.png")
image3 = PhotoImage(file=path + "/plant_segmented_image.png")
image4 = PhotoImage(file=path + "/segmented_forest_image.png")


# Labels für die Kacheln erstellen und Bilder zuweisen
label1 = tk.Label(frame1, image=image1)
label1.grid()
label2 = tk.Label(frame2, image=image2)
label2.grid()
label3 = tk.Label(frame3, image=image3)
label3.grid()
label4 = tk.Label(frame4, image=image4)
label4.grid()

# Text für die Kacheln erstellen
text1 = tk.Label(frame1, text="Satelliten-Aufnahme",fg="black", background='white')
text1.grid()

text2 = tk.Label(frame2, text=percent_white_black("street") + " % der Aufnahme sind Straßen", fg="black", background='white')
text2.grid()

text3 = tk.Label(frame3, text=percent_white_black("plants") + " % der Aufnahme sind mit Pflanzen bewachsen", fg="black", background='white')
text3.grid()

text4 = tk.Label(frame4, text=percent_white_black("forest") + " % der Aufnahme sind Wälder", background='white', fg="black")
text4.grid()

third.mainloop()