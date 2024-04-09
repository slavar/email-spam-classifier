import pickle


class Classifier:

    def __init__(self, serialized_model='app/model.pkl', serialized_tfidf='app/tf.pkl'):
        # deserialize Tf-Idf Vector & Model
        with open(serialized_tfidf, 'rb') as f_tf:
            self.tf = pickle.load(f_tf)

        with open(serialized_model, 'rb') as f_m:
            self.model = pickle.load(f_m)

    def predict(self, message):
        message_vector = self.tf.transform(message)
        category = self.model.predict(message_vector)

        return category
