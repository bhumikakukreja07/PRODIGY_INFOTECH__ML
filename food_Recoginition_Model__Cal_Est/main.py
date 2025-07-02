from fastai.vision.all import *

def main():
    dls = ImageDataLoaders.from_folder(
        'data/food-101-tiny',
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    )
    
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(3)
    # learn.export('export.pkl')
    learn.save('model_weights')  # safer alternative to export.pkl

if __name__ == '__main__':
    main()