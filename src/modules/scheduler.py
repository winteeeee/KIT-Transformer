import tensorflow as tf


class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model,):
        super(TransformerScheduler, self).__init__()


    def __call__(self, step):
        pass

    def get_config(self):
        pass
