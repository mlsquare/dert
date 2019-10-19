import numpy as np
from keras.models import model_from_json
from flask_api import FlaskAPI
from flask import request

app = FlaskAPI(__name__)

chars = ['S', 'L', 'R', 'E']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")

    loaded_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model


path_model = load_model('path_model')

path_model.summary()


def get_outputs(x_latent, sent_latent=[1, 0, 0, 0], index=0):
    print(np.array(x_latent))
    print(np.array(x_latent).shape)
    x_latent = np.array(x_latent).reshape(1, 5)
    x_sent = np.zeros((1, 7, 4))
    x_sent[0, index, np.argmax(sent_latent)] = 1
    pred = path_model.predict([x_latent, x_sent])
    char_index = np.argmax(pred[0])
    sent_latent = [0, 0, 0, 0]
    sent_latent[char_index] = 1
    next_char = indices_char[char_index]
    if next_char == 'E': # or max len -- check path_model summary 
        print('Reached a leaf node.')
        return sent_latent, pred[0], next_char
    return sent_latent, pred[0], next_char


# response = get_outputs([1.0, -0.95, -0.91, 0.99, 0.29], [0, 0, 1, 0], 6)

path_progress = {
    0: {
        'step': 'S'
    }
}

@app.route("/", methods=['GET', 'POST'])
def pes_project():
    if request.method == 'POST':
        # x_latent = request.data.get('feature_embedding')
        print(request)
        # outputs = get_outputs(x_latent)
        return request
        # return outputs, status.HTTP_201_CREATED

    # request.method == 'GET'
    return path_progress

if __name__ == "__main__":
    app.run(debug=True)

