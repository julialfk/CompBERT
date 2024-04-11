from transformers import pipeline

def main():
    classifier = pipeline("sentiment-analysis", model="microsoft/codebert-base")
    print(classifier("We are very happy to show you the 🤗 Transformers library."))

if __name__ == "__main__":
    main()