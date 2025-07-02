from fastai.vision.all import *
import json

# Load calorie dictionary
with open('calories.json') as f:
    calories = json.load(f)

# Rebuild model and load weights
dls = ImageDataLoaders.from_folder('data/food-101-tiny', valid_pct=0.2, item_tfms=Resize(224))
learn = vision_learner(dls, resnet18)
learn.load('model_weights')  # make sure you saved with learn.save('model_weights')

# Predict
img = PILImage.create('data/food-101-tiny/train/ramen/1603.jpg')
pred, _, _ = learn.predict(img)
print(f"{pred}: {calories.get(str(pred), 'Unknown')} kcal")