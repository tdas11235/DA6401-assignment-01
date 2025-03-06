import data_loader as dl
import trainers as tr


(x_train, y_train), (x_test, y_test) = dl.load_fashion_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
tr.normal_trainer(x_train, y_train, 784, 10, [
                  64, 32, 32], epochs=10, act_type='sigmoid', optimiser_type='GD', lr=0.1, loss_type='ce')
