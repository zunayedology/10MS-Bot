from rag_pipeline import setup_rag_pipeline

def evaluate_rag():
    rag_fn = setup_rag_pipeline()
    if not rag_fn:
        print("Pipeline setup failed.")
        return 0.0

    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
    ]

    correct = 0
    for test in test_cases:
        try:
            answer = rag_fn(test["question"])
            print(f"\nQ: {test['question']}")
            print(f"Expected: {test['expected']}")
            print(f"Got: {answer}")
            if test["expected"] in answer:
                correct += 1
        except Exception as e:
            print(f"Error on {test['question']}: {e}")

    accuracy = correct / len(test_cases)
    print(f"\nAccuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    evaluate_rag()
