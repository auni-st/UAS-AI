from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

expression_dict = {0: "MARAH", 1: "JIJIK", 2: "TAKUT", 3: "SENANG", 4: "NETRAL", 5: "SEDIH", 6: "TERKEJUT"}

json_file = open('expression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
expression_model = model_from_json(loaded_model_json)

expression_model.load_weights("expression_model.h5")
print("Model berhasil dimuat")

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=15,
        color_mode="grayscale",
        class_mode='categorical')

predictions = expression_model.predict_generator(test_generator)

print("-----------------------------------------------------------------")
print("CLASSIFICATION REPORT")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))




