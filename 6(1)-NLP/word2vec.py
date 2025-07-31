import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        pad_token_id = tokenizer.pad_token_id

        if self.method == "cbow":
            self._train_cbow(corpus, tokenizer, criterion, optimizer, num_epochs, pad_token_id)
        elif self.method == "skipgram":
            self._train_skipgram(corpus, tokenizer, criterion, optimizer, num_epochs, pad_token_id)
        else:
            assert False

    def _train_cbow(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        criterion,
        optimizer,
        num_epochs: int,
        pad_token_id: int
    ) -> None:
        # CBOW: context -> center
        for epoch in range(num_epochs):
            total_loss = 0
            for sentence in corpus:
                token_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
                for i in range(self.window_size, len(token_ids) - self.window_size):
                    center = token_ids[i]
                    if center == pad_token_id:
                        continue
                    context = [
                        token_ids[j]
                        for j in range(i - self.window_size, i + self.window_size + 1)
                        if j != i and token_ids[j] != pad_token_id
                    ]
                    if not context:
                        continue
                    context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # (1, context_size)
                    center_tensor = torch.tensor([center], dtype=torch.long)  # (1,)
                    optimizer.zero_grad()
                    context_embeds = self.embeddings(context_tensor).mean(dim=1)  # (1, d_model)
                    logits = self.weight(context_embeds)  # (1, vocab_size)
                    loss = criterion(logits, center_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        criterion,
        optimizer,
        num_epochs: int,
        pad_token_id: int
    ) -> None:
        # Skip-gram: center -> context
        for epoch in range(num_epochs):
            total_loss = 0
            for sentence in corpus:
                token_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
                for i in range(self.window_size, len(token_ids) - self.window_size):
                    center = token_ids[i]
                    if center == pad_token_id:
                        continue
                    context = [
                        token_ids[j]
                        for j in range(i - self.window_size, i + self.window_size + 1)
                        if j != i and token_ids[j] != pad_token_id
                    ]
                    if not context:
                        continue
                    center_tensor = torch.tensor([center], dtype=torch.long)  # (1,)
                    for target in context:
                        target_tensor = torch.tensor([target], dtype=torch.long)  # (1,)
                        optimizer.zero_grad()
                        center_embed = self.embeddings(center_tensor)  # (1, d_model)
                        logits = self.weight(center_embed)  # (1, vocab_size)
                        loss = criterion(logits, target_tensor)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
