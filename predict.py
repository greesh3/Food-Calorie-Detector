from roboflow import Roboflow
rf = Roboflow(api_key="gRQVg1b6iNDDd7KB2m7Q")
project = rf.workspace().project("indian_food_dataset")
model = project.version(4).model

# infer on a local image
import pandas as pd

dataFrame = pd.read_csv("C:\\Users\\fathi\\OneDrive\\Desktop\\major_project\\lookup.csv")
#print(dataFrame)
import json

di = model.predict("C:\\Users\\fathi\\OneDrive\\Desktop\\CalorieDetector\\train\\images\\94_jpg.rf.a4b6f032d71e7bda1efc0f25ff50ecdc.jpg", confidence=40, overlap=30).json()

cl = (di['predictions'][0]['class'])
print(cl)
if cl == 'Appam':
    conditional_row = dataFrame.iloc[(dataFrame['Label'] == 'Appam').values]
    print(conditional_row)
elif cl == 'Apple':
    conditional_row = dataFrame.iloc[(dataFrame['Label'] == 'Apple').values]
    print(conditional_row)
elif cl == 'Aloogobi':
    conditional_row = dataFrame.iloc[(dataFrame['Label'] == 'Aloogobi').values]
    print(conditional_row)
elif cl == 'Avial':
    conditional_row = dataFrame.iloc[(dataFrame['Label'] == 'Aloogobi').values]
    print(conditional_row)
#visualize your prediction
model.predict("C:\\Users\\fathi\\OneDrive\\Desktop\\CalorieDetector\\train\\images\\94_jpg.rf.a4b6f032d71e7bda1efc0f25ff50ecdc.jpg", confidence=40, overlap=30).save("prediction.jpg")
