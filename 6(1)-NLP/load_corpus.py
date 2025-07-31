from datasets import load_dataset

def load_corpus() -> list[str]:
    """
    Load a corpus for Word2Vec training from 'google-research-datasets/poem_sentiment'
    """
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    
    corpus: list[str] = []
    
    # Extract text from the dataset
    for example in dataset:
        text = example["verse_text"].strip()
        # Only add non-empty text
        if text:
            corpus.append(text)

    # for fast local training
    # corpus = corpus[:1000]
    
    return corpus
