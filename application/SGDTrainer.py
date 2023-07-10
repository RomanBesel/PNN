class SGDTrainer(object):

    batch_size = 1
    alpha = 0.05
    amount_epochs = 1
    shuffle = False

    def __init__(self, batch_size, alpha, amount_epochs, shuffle):
        self.batch_size = batch_size
        self.alpha = alpha
        self.amount_epochs = amount_epochs
        self.shuffle = shuffle

    def optimize(self, network, data):
        for i in range(self.amount_epochs):
            for j in range(len(data[0])):

                single_data = [data[0][j], data[1][j]]
                network.backprop(single_data, self.alpha)

        return network
