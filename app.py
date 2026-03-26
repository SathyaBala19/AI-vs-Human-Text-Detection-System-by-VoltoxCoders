# i built the interactive demo part of the project
# user can type any text and it will classify it as human or AI

import os
import sys
import argparse
import textwrap
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.decision_layer import decide

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")

model_options = {
    "lr": {
        "vec": os.path.join(models_dir, "tfidf_vectorizer.joblib"),
        "clf": os.path.join(models_dir, "logistic_regression.joblib"),
        "name": "Logistic Regression"
    },
    "nb": {
        "vec": os.path.join(models_dir, "tfidf_vectorizer.joblib"),
        "clf": os.path.join(models_dir, "naive_bayes.joblib"),
        "name": "Naive Bayes"
    }
}

# i wrote these sample texts myself to test different scenarios in the demo
demo_samples = [
    {
        "info": "Human written - student essay",
        "text": (
            "i think driverless cars are a cool idea but im not totally sure they "
            "are safe yet. like what happens when theres a really bad storm and the "
            "sensors cant see properly. my dad says the technology isnt ready and i "
            "kinda agree with him on that."
        )
    },
    {
        "info": "AI generated - formal style",
        "text": (
            "Autonomous vehicles represent a significant technological advancement in "
            "modern transportation. These systems leverage machine learning algorithms, "
            "sensor fusion, and real-time data processing to navigate complex environments "
            "while adhering to regulatory frameworks and cybersecurity protocols."
        )
    },
    {
        "info": "Human written - casual blog post",
        "text": (
            "so i tried making sourdough last weekend and it was a total disaster lol "
            "the starter looked okay but when i baked it the loaf came out super dense. "
            "my roommate said it tasted like cardboard which honestly fair. gonna try "
            "again next weekend after watching some youtube videos."
        )
    },
    {
        "info": "AI generated - more structured",
        "text": (
            "Climate change refers to long-term shifts in global temperatures and weather "
            "patterns. Scientific evidence indicates that human activities, particularly "
            "the burning of fossil fuels, have been the dominant driver since the mid-20th "
            "century. Mitigation requires transitions in energy production and transportation."
        )
    },
    {
        "info": "Borderline - could go either way",
        "text": (
            "Exercise is really important for staying healthy. It helps maintain a healthy "
            "weight and also improves your mood a lot. I try to walk every day after school "
            "even when im tired because i know its good for me in the long run."
        )
    }
]


def load_model(key):
    info = model_options[key]
    if not os.path.exists(info["clf"]):
        print("\n ERROR: model files not found. Run python src/train.py first.\n")
        sys.exit(1)
    vec = joblib.load(info["vec"])
    clf = joblib.load(info["clf"])
    return vec, clf, info["name"]


def run_prediction(text, vec, clf):
    if not text.strip():
        print("  no text entered, please type something.")
        return
    cleaned = clean_text(text)
    decide(clf, vec, cleaned)


def run_demo(vec, clf, name):
    print("\n running demo samples with", name, "\n")
    for i, s in enumerate(demo_samples, 1):
        print("  [" + str(i) + "]", s["info"])
        wrapped = textwrap.fill(s["text"], width=65,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        run_prediction(s["text"], vec, clf)
        input("  press Enter to continue...")
        print()


def interactive_mode(vec, clf, name):
    print("\n" + "=" * 55)
    print("  AI vs HUMAN TEXT DETECTION  -  Interactive Mode")
    print("=" * 55)
    print("  model:", name)
    print("  commands: :model lr | :model nb | :demo | :quit")
    print("  paste or type text then press Enter twice")
    print("=" * 55 + "\n")

    while True:
        print("  Enter text:")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break

                cmd = line.strip().lower()

                if cmd in (":quit", "quit", "q", "exit"):
                    print("\n  Bye!\n")
                    sys.exit(0)

                elif cmd == ":demo":
                    run_demo(vec, clf, name)
                    lines = []
                    break

                elif cmd.startswith(":model "):
                    chosen = cmd.split()[1]
                    if chosen not in model_options:
                        print("  unknown model. use lr or nb")
                    else:
                        vec, clf, name = load_model(chosen)
                        print("  switched to", name)
                    lines = []
                    break

                else:
                    lines.append(line)

        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!\n")
            sys.exit(0)

        if lines:
            run_prediction(" ".join(lines), vec, clf)


def main():
    parser = argparse.ArgumentParser(description="AI vs Human Text Detection")
    parser.add_argument("--model", choices=["lr", "nb"], default="lr",
                        help="lr = Logistic Regression (default), nb = Naive Bayes")
    parser.add_argument("--text", type=str, default=None,
                        help="give a text directly without interactive mode")
    args = parser.parse_args()

    vec, clf, name = load_model(args.model)

    if args.text:
        print("\n  model:", name)
        run_prediction(args.text, vec, clf)
    else:
        interactive_mode(vec, clf, name)


if __name__ == "__main__":
    main()
