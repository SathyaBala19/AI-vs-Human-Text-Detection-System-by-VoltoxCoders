# my part - the decision layer
# once the model gives a probability score we need to convert that into
# something meaningful for the user
# instead of just saying "AI" or "Human" we show how confident the model is
# and give a final verdict based on that

# i picked these threshold values after looking at the model results
# both models were very accurate (LR ~99%, NB ~97%) and most predictions
# had confidence above 0.90 so 0.85 felt like a fair line for "high certainty"
# anything below 0.60 means the model is barely leaning one way - better to flag those

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.60


def get_verdict(conf):
    if conf >= HIGH_THRESHOLD:
        return "Acceptable - High Certainty"
    elif conf >= LOW_THRESHOLD:
        return "Needs Review - Moderate Certainty"
    else:
        return "Uncertain / Likely AI-generated - Low Certainty"


def get_tier(conf):
    if conf >= HIGH_THRESHOLD:
        return 1
    elif conf >= LOW_THRESHOLD:
        return 2
    return 3


def show_result(label, conf, p_human, p_ai):
    print("")
    print("=" * 48)
    print("   AI vs HUMAN TEXT DETECTION - RESULT")
    print("=" * 48)
    print("  Prediction   :", label)
    print("  Confidence   :", str(round(conf * 100, 2)) + "%")
    print("  P(Human)     :", round(p_human, 4))
    print("  P(AI)        :", round(p_ai, 4))
    print("-" * 48)
    print("  Final Decision :", get_verdict(conf))
    print("  Tier           :", get_tier(conf))
    print("=" * 48)
    print("")


def decide(model, vectorizer, cleaned_text):
    X = vectorizer.transform([cleaned_text])

    # index 0 = Human, index 1 = AI generated
    probs = model.predict_proba(X)[0]
    p_human = float(probs[0])
    p_ai = float(probs[1])

    pred = model.predict(X)[0]

    if pred == 0:
        label = "Human"
        conf = p_human
    else:
        label = "AI-generated"
        conf = p_ai

    show_result(label, conf, p_human, p_ai)

    return {
        "label": label,
        "conf": conf,
        "p_human": p_human,
        "p_ai": p_ai,
        "verdict": get_verdict(conf),
        "tier": get_tier(conf)
    }
