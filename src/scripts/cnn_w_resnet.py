from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from engine.export import load_audio, metadata_db

"""Tentative to use the RESNet model, code doesn't work"""


MODEL_NAME = 'cnn_w_resnet'
WIDTH = 224
HEIGHT = 224
NUM_CLASSES = 2
LEARNING_RATE = .0001
MAX_EPOCHS = 50

mdb = metadata_db('metadata.json')

base = (
    load_audio(mdb, 'w_1718_train.txt').create_spectrogram(height=HEIGHT, width=WIDTH).img_features()
)

resNet: Model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

x = resNet.output

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=resNet.input, outputs=x)

for l in resNet.layers:
    l.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['acc'])

model.summary(line_length=100)

print(model.output_shape)

m = base.create_nn_model(MODEL_NAME, model, prepare_args={'triplicate': True, 'kiu_preprocess': True})

m.train(MAX_EPOCHS).run()

acc = []
val = load_audio(mdb, 'w_1718_val.txt').preproc_from(m)
for i in trange(1, MAX_EPOCHS + 1):
    r = m.train(i).recognize(val).extract().get()
    acc.append(r["accuracy"])

print('\n'.join(f'{e}: {acc:.2%}' for e, acc in enumerate(acc)))
