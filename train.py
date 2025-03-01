import data_loader as dl
import trainers as tr


(x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
tr.normal_trainer(x_train, y_train, 784, 10, [
                  128, 128, 64], epochs=10, act_type='relu', optimiser_type='GD', lr=0.1, loss_type='mse')
