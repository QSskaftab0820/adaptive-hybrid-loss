import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    # 1. Get prediction probabilities
    y_prob = model.predict(X_test, verbose=0)

    # 2. Convert probabilities to class labels
    y_pred = (y_prob > 0.5).astype(int)

    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 4. Create results folder if not exists
    os.makedirs("results", exist_ok=True)

    # 5. Save metrics to metrics.txt
    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1-score: {f1}\n")
        f.write(f"AUC: {auc}\n")

    print("Metrics saved to results/metrics.txt")

    # 6. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()

    print("ROC curve saved to results/roc_curve.png")

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved to results/confusion_matrix.png")

    # 8. Print metrics on screen (for screenshots)
    print("\nFinal Evaluation Metrics")
    print("------------------------")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("AUC      :", auc)
