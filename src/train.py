from sklearn.metrics import f1_score
from loss_functions import AdaptiveHybridLoss
from model import build_model

def update_weights(f1_score):
    if f1_score < 0.7:
        return 0.3, 0.5, 0.2
    elif f1_score < 0.8:
        return 0.4, 0.4, 0.2
    else:
        return 0.5, 0.3, 0.2

def train_model(X_train, y_train, X_val, y_val):
    loss_obj = AdaptiveHybridLoss()
    model = build_model(X_train.shape[1])

    model.compile(optimizer='adam', loss=loss_obj, metrics=['accuracy'])

    for epoch in range(10):
        model.fit(X_train, y_train, epochs=1, batch_size=256, verbose=1)

        y_pred = (model.predict(X_val) > 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)

        a, b, c = update_weights(f1)
        loss_obj.alpha, loss_obj.beta, loss_obj.gamma = a, b, c

    return model

if __name__ == "__main__":
    from utils import load_data
    from evaluate import evaluate_model

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
