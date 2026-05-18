# evaluate_chatbot.py
# ✅ Confusion Matrix for Diksha Chatbot
# ✅ Language Detection Accuracy
# ✅ Out-of-scope Classifier Accuracy
# ✅ Answer Quality Evaluation
# ✅ Full Report generate karta hai

import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    import matplotlib
    matplotlib.use("Agg")  # No display needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  sklearn/matplotlib not installed — install with:")
    print("    pip install scikit-learn matplotlib seaborn")
    SKLEARN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# TEST DATASET
# ══════════════════════════════════════════════════════════════════════

# ── Language Detection Tests ──────────────────────────────────────────
LANG_TESTS = [
    # (question, expected_lang)
    # English
    ("What are the fees for BTech?",                    "en"),
    ("Who is the director of GBPIET?",                  "en"),
    ("Tell me about hostel facilities",                 "en"),
    ("What courses are offered?",                       "en"),
    ("How to reach GBPIET?",                            "en"),
    ("Who is the HOD of CSE?",                          "en"),
    ("What is the placement record?",                   "en"),
    ("Contact number of registrar",                     "en"),

    # Hindi
    ("जीबीपीआईईटी की फीस कितनी है?",                  "hi"),
    ("निदेशक कौन हैं?",                                "hi"),
    ("हॉस्टल की सुविधाएं क्या हैं?",                  "hi"),
    ("कौन से कोर्स उपलब्ध हैं?",                      "hi"),
    ("प्लेसमेंट का रिकॉर्ड क्या है?",                 "hi"),
    ("रजिस्ट्रार का नंबर क्या है?",                    "hi"),
    ("CSE के HOD कौन हैं?",                            "hi"),
    ("admission kaise hota hai",                        "hi"),

    # Garhwali
    ("फीस कति छ?",                                     "ga"),
    ("एडमिशन कनकै होंद?",                              "ga"),
    ("हॉस्टल माँ कि सुविधा छ?",                        "ga"),
    ("निदेशक को छ?",                                    "ga"),

    # Kumauni
    ("फीस कतु छु?",                                    "ku"),
    ("एडमिशन कसि होंछ?",                               "ku"),
    ("हॉस्टल मा कतु सुबिद छ?",                         "ku"),
    ("निदेशक को छु?",                                   "ku"),
]

# ── Out-of-scope Tests ────────────────────────────────────────────────
SCOPE_TESTS = [
    # (question, expected: "in_scope" or "out_scope")
    # IN SCOPE — college related
    ("What are the fees for BTech?",                    "in_scope"),
    ("Who is the HOD of CSE?",                          "in_scope"),
    ("Tell me about hostel facilities",                 "in_scope"),
    ("admission process kya hai",                       "in_scope"),
    ("placement record GBPIET",                         "in_scope"),
    ("director kaun hai",                               "in_scope"),
    ("library timing kya hai",                          "in_scope"),
    ("transport facility hai kya",                      "in_scope"),
    ("anti ragging policy",                             "in_scope"),
    ("scholarship kaise milti hai",                     "in_scope"),

    # OUT OF SCOPE — non-college
    ("What is the IPL score today?",                    "out_scope"),
    ("Salman Khan ki film kaunsi hai?",                 "out_scope"),
    ("Aaj ka mausam kaisa hai?",                        "out_scope"),
    ("Bitcoin price kya hai?",                          "out_scope"),
    ("Modi ji ke baare mein batao",                     "out_scope"),
    ("Taj Mahal kahan hai?",                            "out_scope"),
    ("Recipe of biryani",                               "out_scope"),
    ("Cricket match result",                            "out_scope"),
    ("Stock market update",                             "out_scope"),
    ("Bollywood news today",                            "out_scope"),
]

# ── Answer Quality Tests ──────────────────────────────────────────────
ANSWER_TESTS = [
    # (question, lang, expected_keywords_in_answer)
    ("What are the fees for BTech?",    "en", ["fees", "fee", "tuition", "per year", "semester"]),
    ("Who is the director?",            "en", ["director", "GBPIET", "dr", "prof"]),
    ("Tell me about hostel",            "en", ["hostel", "accommodation", "facility", "room"]),
    ("What courses are offered?",       "en", ["btech", "mtech", "mca", "course", "programme"]),
    ("fees kitni hai",                  "hi", ["फीस", "शुल्क", "हजार", "लाख", "प्रति"]),
    ("director kaun hai",               "hi", ["निदेशक", "GBPIET", "डॉ", "प्रो"]),
    ("hostel ki jankari do",            "hi", ["हॉस्टल", "छात्रावास", "सुविधा", "कमरा"]),
    ("placement record kya hai",        "hi", ["प्लेसमेंट", "कंपनी", "पैकेज", "नियुक्ति"]),
]


# ══════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def evaluate_language_detection():
    """Test language_detector.py accuracy."""
    print("\n" + "="*60)
    print("TEST 1: LANGUAGE DETECTION ACCURACY")
    print("="*60)

    try:
        from language_detector import detect_language
    except ImportError:
        print("❌ language_detector not found — skipping")
        return None, None

    actual    = []
    predicted = []
    errors    = []

    for question, expected_lang in LANG_TESTS:
        detected = detect_language(question)
        actual.append(expected_lang)
        predicted.append(detected)

        status = "✅" if detected == expected_lang else "❌"
        if detected != expected_lang:
            errors.append({
                "question": question,
                "expected": expected_lang,
                "got":      detected,
            })
        print(f"{status} [{expected_lang}→{detected}] {question[:50]}")

    print(f"\nTotal: {len(LANG_TESTS)} | Correct: {len(LANG_TESTS)-len(errors)} | Wrong: {len(errors)}")
    print(f"Accuracy: {(len(LANG_TESTS)-len(errors))/len(LANG_TESTS)*100:.1f}%")

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  Q: {e['question'][:50]}")
            print(f"  Expected: {e['expected']} | Got: {e['got']}")

    return actual, predicted


def evaluate_scope_detection():
    """Test out-of-scope detector accuracy."""
    print("\n" + "="*60)
    print("TEST 2: OUT-OF-SCOPE DETECTION ACCURACY")
    print("="*60)

    try:
        from rag.kb_query import is_out_of_scope
    except ImportError:
        print("❌ kb_query not found — skipping")
        return None, None

    actual    = []
    predicted = []
    errors    = []

    for question, expected in SCOPE_TESTS:
        is_oos   = is_out_of_scope(question)
        detected = "out_scope" if is_oos else "in_scope"

        actual.append(expected)
        predicted.append(detected)

        status = "✅" if detected == expected else "❌"
        if detected != expected:
            errors.append({
                "question": question,
                "expected": expected,
                "got":      detected,
            })
        print(f"{status} [{expected}→{detected}] {question[:50]}")

    print(f"\nTotal: {len(SCOPE_TESTS)} | Correct: {len(SCOPE_TESTS)-len(errors)} | Wrong: {len(errors)}")
    print(f"Accuracy: {(len(SCOPE_TESTS)-len(errors))/len(SCOPE_TESTS)*100:.1f}%")

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  Q: {e['question'][:50]}")
            print(f"  Expected: {e['expected']} | Got: {e['got']}")

    return actual, predicted


def evaluate_answer_quality():
    """Test answer quality — keywords present in answer."""
    print("\n" + "="*60)
    print("TEST 3: ANSWER QUALITY")
    print("="*60)

    try:
        from rag.kb_query import get_answer
    except ImportError:
        print("❌ kb_query not found — skipping")
        return None, None

    actual    = []
    predicted = []
    results   = []

    for question, lang, keywords in ANSWER_TESTS:
        print(f"\n[{lang}] {question}")
        try:
            answer = get_answer(question, lang)
            answer_lower = answer.lower()

            # Check if any keyword found in answer
            found = any(kw.lower() in answer_lower for kw in keywords)
            quality = "correct" if found else "wrong"

            actual.append("correct")
            predicted.append(quality)

            status = "✅" if found else "❌"
            print(f"{status} Answer: {answer[:100]}...")
            if not found:
                print(f"   Expected keywords: {keywords}")

            results.append({
                "question": question,
                "lang":     lang,
                "answer":   answer[:200],
                "quality":  quality,
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            actual.append("correct")
            predicted.append("no_answer")

    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    print(f"\nTotal: {len(ANSWER_TESTS)} | Correct: {correct} | Wrong: {len(ANSWER_TESTS)-correct}")
    print(f"Answer Quality: {correct/len(ANSWER_TESTS)*100:.1f}%")

    return actual, predicted


# ══════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(actual, predicted, labels, title, filename):
    """Generate and save confusion matrix plot."""
    if not SKLEARN_AVAILABLE:
        print(f"⚠️  Skipping plot: {filename}")
        return

    cm = confusion_matrix(actual, predicted, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {filename}")


def print_metrics(actual, predicted, labels, title):
    """Print precision, recall, f1 for each class."""
    if not SKLEARN_AVAILABLE:
        return

    print(f"\n{'─'*40}")
    print(f"Metrics — {title}")
    print(f"{'─'*40}")
    print(classification_report(actual, predicted, labels=labels, zero_division=0))

    overall_acc = accuracy_score(actual, predicted)
    print(f"Overall Accuracy: {overall_acc*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════

def generate_report(results: dict):
    """Save JSON report."""
    report = {
        "timestamp":   datetime.now().isoformat(),
        "chatbot":     "Diksha — GBPIET",
        "results":     results,
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n✅ Report saved: evaluation_report.json")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DIKSHA CHATBOT — EVALUATION WITH CONFUSION MATRIX")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # ── Test 1: Language Detection ────────────────────────────────────
    lang_actual, lang_predicted = evaluate_language_detection()

    if lang_actual and lang_predicted and SKLEARN_AVAILABLE:
        labels = ["en", "hi", "ga", "ku"]
        plot_confusion_matrix(
            lang_actual, lang_predicted,
            labels=labels,
            title="Language Detection — Confusion Matrix",
            filename="cm_language_detection.png",
        )
        print_metrics(lang_actual, lang_predicted, labels, "Language Detection")

        results["language_detection"] = {
            "accuracy": accuracy_score(lang_actual, lang_predicted),
            "total":    len(lang_actual),
            "correct":  sum(a == p for a, p in zip(lang_actual, lang_predicted)),
        }

    # ── Test 2: Scope Detection ───────────────────────────────────────
    scope_actual, scope_predicted = evaluate_scope_detection()

    if scope_actual and scope_predicted and SKLEARN_AVAILABLE:
        labels = ["in_scope", "out_scope"]
        plot_confusion_matrix(
            scope_actual, scope_predicted,
            labels=labels,
            title="Out-of-Scope Detection — Confusion Matrix",
            filename="cm_scope_detection.png",
        )
        print_metrics(scope_actual, scope_predicted, labels, "Scope Detection")

        results["scope_detection"] = {
            "accuracy":  accuracy_score(scope_actual, scope_predicted),
            "precision": precision_score(scope_actual, scope_predicted, pos_label="out_scope", zero_division=0),
            "recall":    recall_score(scope_actual, scope_predicted, pos_label="out_scope", zero_division=0),
            "f1":        f1_score(scope_actual, scope_predicted, pos_label="out_scope", zero_division=0),
            "total":     len(scope_actual),
            "correct":   sum(a == p for a, p in zip(scope_actual, scope_predicted)),
        }

    # ── Test 3: Answer Quality ────────────────────────────────────────
    ans_actual, ans_predicted = evaluate_answer_quality()

    if ans_actual and ans_predicted and SKLEARN_AVAILABLE:
        labels = ["correct", "wrong", "no_answer"]
        plot_confusion_matrix(
            ans_actual, ans_predicted,
            labels=labels,
            title="Answer Quality — Confusion Matrix",
            filename="cm_answer_quality.png",
        )
        print_metrics(ans_actual, ans_predicted, labels, "Answer Quality")

        results["answer_quality"] = {
            "accuracy": accuracy_score(ans_actual, ans_predicted),
            "total":    len(ans_actual),
            "correct":  sum(a == p for a, p in zip(ans_actual, ans_predicted)),
        }

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for test_name, data in results.items():
        acc = data.get("accuracy", 0) * 100
        print(f"{test_name:30s} → Accuracy: {acc:.1f}%  ({data['correct']}/{data['total']})")

    if SKLEARN_AVAILABLE:
        print("\n📊 Confusion matrix images saved:")
        print("   cm_language_detection.png")
        print("   cm_scope_detection.png")
        print("   cm_answer_quality.png")

    generate_report(results)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
