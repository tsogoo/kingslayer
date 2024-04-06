# kingslayer


# generate chessboard training data

copy chess.blend file on `blender` folder and run:

`patto/blender.exe pathto\chess.blend -b --python pathto\generate_chessboard_data.py`

to train chessboard via yolov8:

`python train.py`

show graphs:

`tensorboard --logdir runs/detect/train/' or 'tensorboard --logdir runs/detect/train{Index}/`

to run prediction:

`python predict.py --weights=runs/train/weights/best.pt --image=datasets/test/images/0000.png`

if prediction ran success then result would be in `runs/predict/0000.png`