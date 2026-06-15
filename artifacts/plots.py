# plots.py
import os, joblib, numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

ARTIFACT_DIR = "./artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_model.pkl")

def plot_confusion_matrix(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_score, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC={auc:.4f})")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_feature_importance(model, outpath, top_k=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_k:]
    vals = importances[idx]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.barh(range(top_k), vals)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"f_{i}" for i in idx])
    ax.set_xlabel("Importance"); ax.set_title(f"Top {top_k} Features (RF)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    X_test  = load_npz(os.path.join(ARTIFACT_DIR, "X_test.npz"))
    y_test  = np.load(os.path.join(ARTIFACT_DIR, "y_test.npy"))
    rf = joblib.load(MODEL_PATH)

    # Confusion matrix + ROC
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ["Normal(0)", "Attack(1)"], os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))
    plot_roc(y_test, y_proba, os.path.join(ARTIFACT_DIR, "roc_curve.png"))

    # Feature importance (index names; SHAP gives semantic names)
    plot_feature_importance(rf, os.path.join(ARTIFACT_DIR, "rf_feature_importance.png"))

    # SHAP summary plot (optional)
    try:
        import shap
        vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"))
        # small slice to keep it light
        Xs = X_test[:400].toarray()
        explainer = shap.TreeExplainer(rf, feature_perturbation="interventional")
        shap_values = explainer.shap_values(Xs, check_additivity=False)
        shap.summary_plot(shap_values[1], Xs, show=False)
        plt.gcf().savefig(os.path.join(ARTIFACT_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[Info] SHAP plot skipped:", e)

if __name__ == "__main__":
    main()
