from lime.lime_text import LimeTextExplainer
import joblib


class_names = ["negative", "positive"]


def load_model():
    model = joblib.load("sentiment_model.pkl")
    return model


def explain_with_lime(model, text: str):
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        return model.predict_proba(texts)

    true_label = model.predict([text])[0]

    print("\n====================")
    print("Text:")
    print(text)
    print("--------------------")
    print("Model prediction:", class_names[true_label])
    print("====================\n")

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=5,
        labels=[true_label],
    )

    print("Top words contributing to prediction:")
    for word, weight in exp.as_list(label=true_label):
        print(f"{word:20s}  weight={weight:.4f}")


if __name__ == "__main__":
    model = load_model()
    text = "I really loved the movie, it was fantastic!"
    explain_with_lime(model, text)
