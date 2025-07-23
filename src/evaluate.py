from rag_pipeline import setup_rag_pipeline

def evaluate_rag():
    rag_chain = setup_rag_pipeline()
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
    ]
    correct = 0
    for test in test_cases:
        result = rag_chain({"question": test["question"]})
        if test["expected"] in result["answer"]:
            correct += 1
    accuracy = correct / len(test_cases)
    print(f"Accuracy (Groundedness): {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    evaluate_rag()