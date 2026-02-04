"""Minimal tokenizer stub for offline unit tests."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class DummyTokenizer:
    """Minimal tokenizer stub covering the interfaces used in tests."""

    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
            "hello": 5,
            "world": 6,
            "short": 7,
            "hi": 8,
            "sample": 9,
            "this": 10,
            "is": 11,
            "text": 12,
            "for": 13,
            "testing": 14,
            "number": 15,
            "with": 16,
            "enough": 17,
            "0": 18,
            "1": 19,
            "2": 20,
            "3": 21,
            "4": 22,
            "5": 23,
            "6": 24,
            "7": 25,
            "8": 26,
            "9": 27,
        }
        self.pad_token_id = self.vocab["[PAD]"]
        self.mask_token_id = self.vocab["[MASK]"]
        self.all_special_ids = [
            self.vocab["[PAD]"],
            self.vocab["[CLS]"],
            self.vocab["[SEP]"],
            self.vocab["[MASK]"],
            self.vocab["[UNK]"],
        ]
        self._unk_id = self.vocab["[UNK]"]

    @property
    def vocab_size(self) -> int:
        """Return vocab size for compatibility with HF tokenizers."""
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def __call__(
        self,
        text: Any,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(text, (list, tuple)):
            input_ids = [
                self._encode(
                    item,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                )
                for item in text
            ]
        else:
            input_ids = self._encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
        return {"input_ids": input_ids}

    def pad(
        self,
        encoded_inputs: Sequence[dict[str, Any]],
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        input_id_list: list[list[int]] = []
        for example in encoded_inputs:
            ids = example["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            input_id_list.append(list(ids))

        max_len = max((len(ids) for ids in input_id_list), default=0)
        padded: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in input_id_list:
            pad_len = max_len - len(ids)
            padded_ids = ids + [self.pad_token_id] * pad_len
            padded.append(padded_ids)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return {"input_ids": padded, "attention_mask": attention_mask}

    def _encode(
        self,
        text: Any,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None,
    ) -> list[int]:
        tokens = str(text).lower().split()
        ids = [self.vocab.get(tok, self._unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.vocab["[CLS]"]] + ids + [self.vocab["[SEP]"]]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids
