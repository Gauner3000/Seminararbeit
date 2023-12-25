from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn



def Neuronal_Network(path, kind):

    
    #Hier ist das Modell definiert:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class UNET(nn.Module):
        def __init__(
                self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
        ):
            super(UNET, self).__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Down part of UNET
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature

            # Up part of UNET
            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                    )
                )
                self.ups.append(DoubleConv(feature*2, feature))

            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skip_connections = []

            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

            return self.final_conv(x)

    # Hier füge ich eine Funktion hinzu, die die Ausgabe des Modells in eine binäre Segmentierung umwandelt.
    def post_process(output, threshold=0.5):
        # Verwendet eine Schwelle (threshold), um die Wahrscheinlichkeiten in 0 und 1 umzuwandeln. (Schwarz und Weiß)
        binary_output = (output > threshold).float()
        return binary_output

    if kind == "street":
        checkpointpath = "street_checkpoint.pth.tar"
        image_save = "/segmented_street_image.png"
    elif kind == "forest":
        checkpointpath = "forest_checkpoint.pth.tar"
        image_save = "/segmented_forest_image.png"


    checkpoint = torch.load(checkpointpath)
    # Erstellt ein neues Modell mit der gleichen Architektur wie das ursprüngliche Modell
    model = UNET(in_channels=3, out_channels=1)  # Passt die Eingabe- und Ausgabekanäle an
    # Lädt das Modellgewicht aus dem Checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    # Setzt das Modell in den Ausführungsmodus (eval mode)
    model.eval()


    ##############################################################################################
    #Bild  wird geladen und die Straßensegmentierung durchgeführt

    # Lädt das Bild
    image = Image.open(path + "/satellite_image.png") #Pfad zum Satelliten-Bild)

    # Wandelt das Bild in das richtige Format um (220x220), transformiert es (selbe Parameter wie beim training) und führt die Bildsegmentierung durch
    preprocess = transforms.Compose([transforms.Resize((220, 220)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                        std=[1.0, 1.0, 1.0])])
    image = preprocess(image).unsqueeze(0)  # Füge eine Batch-Dimension hinzu

    # Führt die Bildsegmentierung durch
    with torch.no_grad():
        output = model(image)
    # Verwendet die post_process-Funktion, um die Ausgabe in eine binäre Segmentierung umzuwandeln
    binary_output = post_process(output)

    # Speichert die binäre Segmentierung in einen Ordner
    output_image = Image.fromarray((binary_output.squeeze().numpy() * 255).astype('uint8'))
    output_image.save(path + image_save)