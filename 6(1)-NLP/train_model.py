import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *


if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    # train
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_sum = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = tokenizer(
                batch["verse_text"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)

            labels = torch.tensor(batch["label"], dtype=torch.long, device=device)

            logits = model(input_ids)         # (B, L, num_classes)

            # ▲ 1) 마지막 토큰 hidden state만 사용 → (B, num_classes)
            if logits.dim() == 3:             # 안전하게 확인
                logits = logits[:, -1, :]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # ─────────────────────────────────────────────────────────────
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in validation_loader:
                input_ids = tokenizer(
                    batch["verse_text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                logits = model(input_ids)     # (B, L, C) → (B, C)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                preds  += logits.argmax(-1).cpu().tolist()
                targets += batch["label"].tolist()
        # ─────────────────────────────────────────────────────────────

        macro = f1_score(targets, preds, average='macro')
        micro = f1_score(targets, preds, average='micro')
        print(f"loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")