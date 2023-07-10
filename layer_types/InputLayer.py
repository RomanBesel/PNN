



class InputLayer():

    def __init__(self):
        print('inputlayer constructed')

    def get_train_data(self, data):
        return data[0]

    def get_label(self, data):
        return data[1]