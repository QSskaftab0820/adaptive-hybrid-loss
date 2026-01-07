import tensorflow as tf

def cross_entropy(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * tf.pow(1 - pt, gamma) * bce
    return loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

class AdaptiveHybridLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal = focal_loss()

    def call(self, y_true, y_pred):
        ce = cross_entropy(y_true, y_pred)
        fl = self.focal(y_true, y_pred)
        dl = dice_loss(y_true, y_pred)
        return self.alpha * ce + self.beta * fl + self.gamma * dl
