import tensorflow as tf
class ModelBuilder:
    def __init__(self,data_function,model = None):
        (x_train,y_train) ,(x_test,y_test) = data_function()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
    def set_model(self,model):
        self.model = model

    def get_flatten_layer(self):
        input_size = self.x_train.shape[1:]
        return tf.keras.layer.Flatten(input_shape=input_size)

    def build(self,
              loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.adam,
              learning_rate = 0.001,
              callbacks = None,
              metrics = None
              ):
        if callbacks is None:
            callbacks = []
        if metrics is None:
            metrics = ['categorical_cross_entropy']
        self.model.compile(
            loss=loss,
            optimizer=optimizer(learning_rate = learning_rate),
            metrics=metrics,
            callbacks = callbacks,
        )
        self.model.fit(self.x_train,self.y_train)
        return self.model

    def get_model(self):
        return self.model