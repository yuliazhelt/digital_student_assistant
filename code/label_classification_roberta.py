from sklearn.preprocessing import MultiLabelBinarizer
from simpletransformers.classification import MultiLabelClassificationModel
from arxiv_categories_preprocess import category_map
import numpy as np

label_classification_model = MultiLabelClassificationModel("roberta", "./outputs", use_cuda=False)
multi_label_encoder = MultiLabelBinarizer()

def predict_category(text):
    predicted_categories_encoded, raw_outputs = label_classification_model.predict([text])
    predicted_categories_encoded = np.array(predicted_categories_encoded)
    predicted_categories_encoded[0][np.argmax(raw_outputs[0])] = 1
    predicted_categories = multi_label_encoder.inverse_transform(predicted_categories_encoded)[0]
    return predicted_categories  # category_map[predicted_categories[0]]