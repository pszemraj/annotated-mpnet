"""
Module containing the logic for handling the dataset for MPNet including the Dataset class as well
as the data collator
"""

import logging
import pathlib
import random
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Sized

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
# NOTE: basicConfig is a no-op if logging is already configured by the host app.
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)


import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizer

from annotated_mpnet.utils import utils

try:
    from annotated_mpnet.utils.perm_utils_fast import make_span_perm
except ImportError:
    make_span_perm = None
    LOGGER.warning(
        "Fast span permutation extension not available; falling back to Python implementation."
    )


def _serialize_np_generator_state(state: Any) -> Any:
    """Convert NumPy generator state into builtin Python types for safe checkpoints.

    :param Any state: NumPy generator state payload.
    :return Any: Serialized state payload.
    """
    if isinstance(state, np.ndarray):
        return state.tolist()
    if isinstance(state, np.generic):
        return state.item()
    if isinstance(state, dict):
        return {k: _serialize_np_generator_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [_serialize_np_generator_state(v) for v in state]
    return state


def _deserialize_np_generator_state(state: Any) -> Any:
    """Return the NumPy generator state payload (best-effort).

    :param Any state: Serialized NumPy generator state.
    :return Any: Deserialized state payload.
    """
    if isinstance(state, dict):
        return {k: _deserialize_np_generator_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_deserialize_np_generator_state(v) for v in state]
    return state


class MPNetDataset(torch.utils.data.Dataset):
    """
    Class handling the collection of samples for MPNet pretraining
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: Optional[str] = None,
        dataset: Optional[Any] = None,
        block_size: int = 512,
        field_name: str = "text",
    ) -> None:
        """Initialize the dataset.

        :param PreTrainedTokenizer tokenizer: The tokenizer for the model.
        :param str file_path: Path to the dataset text file, defaults to None.
        :param object dataset: Pre-loaded HuggingFace dataset or list of examples, defaults to None.
        :param int block_size: Maximum number of tokens per block, defaults to 512.
        :param str field_name: Field containing text in the dataset, defaults to "text".
        :raises ValueError: If neither ``file_path`` nor ``dataset`` is provided.
        :raises ValueError: If ``file_path`` does not exist.
        """
        super().__init__()

        self.tokenizer = tokenizer

        if dataset is not None:
            # Use the provided dataset, which can be either a HF dataset or a list of examples
            if isinstance(dataset, list):
                # For a list of examples
                LOGGER.info(f"Creating features from {len(dataset)} provided examples")
                lines = [example[field_name] for example in dataset]
            else:
                # For a HF dataset
                LOGGER.info(
                    f"Creating features from pre-loaded dataset with {len(dataset)} samples"
                )
                lines = [example[field_name] for example in dataset]
        elif file_path is not None:
            # Check if the file path exists
            if not pathlib.Path(file_path).is_file():
                raise ValueError(f"Input file path {file_path} not found")

            LOGGER.info(f"Creating features from dataset file at {file_path}")

            # We open the file and gather line by line
            with open(file_path, encoding="utf-8") as f:
                lines = [
                    line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())
                ]
        else:
            raise ValueError("Either file_path or dataset must be provided")

        # Process batch encoding using the tokenizer passed in
        batch_encoding = tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=block_size
        )

        # Extract the input IDs and store them
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.examples[i]


class HFStreamingDataset(torch.utils.data.IterableDataset):
    """
    Class for handling streaming datasets from HuggingFace for MPNet pretraining.

    This implements a buffer-based streaming approach to:
    1. Enable efficient streaming of large datasets
    2. Allow for random sampling from the stream
    3. Support multi-worker data loading with appropriate sharding
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: Optional[str] = None,
        dataset_stream: Optional[Any] = None,
        split: str = "train",
        block_size: int = 512,
        buffer_size: int = 10000,
        seed: int = 42,
        min_text_length: int = 200,
        text_field: str = "text",
        skip_samples: int = 0,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize the streaming dataset.

        :param PreTrainedTokenizer tokenizer: The tokenizer for the model.
        :param str dataset_name: HuggingFace dataset name, defaults to None.
        :param object dataset_stream: Pre-loaded streaming dataset, defaults to None.
        :param str split: Dataset split to use, defaults to "train".
        :param int block_size: Maximum number of tokens per block, defaults to 512.
        :param int buffer_size: Buffer size for streaming, defaults to 10000.
        :param int seed: Random seed for reproducibility, defaults to 42.
        :param int min_text_length: Minimum text length to keep, defaults to 200.
        :param str text_field: Field containing text, defaults to "text".
        :param int skip_samples: Number of samples to skip, defaults to 0.
        :param int max_samples: Maximum number of samples to yield, defaults to None.
        :raises ValueError: If neither ``dataset_name`` nor ``dataset_stream`` is provided.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.min_text_length = min_text_length
        self.text_field = text_field
        self.skip_samples = skip_samples
        self.max_samples = max_samples

        # Set random seed
        random.seed(seed)

        # Either use provided stream or load a new one
        if dataset_stream is not None:
            LOGGER.info("Using provided dataset stream")
            # min_text_length filtering is applied only when loading a dataset_name.
            self.dataset = dataset_stream
        elif dataset_name is not None:
            # Load the dataset in streaming mode
            LOGGER.info(f"Loading dataset {dataset_name}, split: {split} in streaming mode")
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)

            # Apply filter for minimum text length if specified
            if min_text_length > 0:
                self.dataset = self.dataset.filter(
                    lambda example: len(example[text_field]) >= min_text_length
                )

            LOGGER.info(f"Dataset {dataset_name} loaded and filtered")
        else:
            raise ValueError("Either dataset_name or dataset_stream must be provided")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over processed text samples.

        :return Iterator[Dict[str, torch.Tensor]]: Iterator yielding processed samples.
        """
        # Set worker seed for proper sharding
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Use different seed for each worker
            worker_seed = worker_info.id + self.seed
            random.seed(worker_seed)

            # Shard the dataset based on worker id
            dataset_iter = iter(
                self.dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            )
        else:
            dataset_iter = iter(self.dataset)

        # skip_samples/max_samples apply per worker when multiple workers are used.
        skip_samples = self.skip_samples
        max_samples = self.max_samples

        # Initialize buffer
        buffer = []

        # Fill the buffer initially
        for _ in range(self.buffer_size):
            try:
                example = next(dataset_iter)
                buffer.append(example)
            except StopIteration:
                break

        # Continue as long as there are items in buffer
        skipped = 0
        yielded = 0
        while buffer:
            # Randomly select an example from the buffer
            idx = random.randint(0, len(buffer) - 1)
            example = buffer[idx]

            # Process the example
            processed = self._process_example(example)

            # Replace the used example with a new one if available
            try:
                buffer[idx] = next(dataset_iter)
            except StopIteration:
                # Remove the used example if no more examples
                buffer.pop(idx)

            if skipped < skip_samples:
                skipped += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                break

            yielded += 1
            yield processed

    def _process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single example from the dataset.

        :param dict example: Dataset example containing text.
        :return Dict[str, torch.Tensor]: Processed input IDs.
        """
        text = example[self.text_field]

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            max_length=self.block_size,
            truncation=True,
        )

        # Leave padding to the collator to avoid redundant max_length padding here.
        return {"input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long)}


def create_hf_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    block_size: int = 512,
    buffer_size: int = 10000,
    seed: int = 42,
    min_text_length: int = 200,
    text_field: str = "text",
    skip_samples: int = 0,
    max_samples: Optional[int] = None,
    collator: Optional[Callable] = None,
    num_workers: int = 4,
) -> DataLoader:
    """Create a dataloader for a HuggingFace dataset in streaming mode.

    :param PreTrainedTokenizer tokenizer: Tokenizer for the model.
    :param str dataset_name: Name of the HuggingFace dataset.
    :param str split: Dataset split to use, defaults to "train".
    :param int batch_size: Batch size for the dataloader, defaults to 32.
    :param int block_size: Maximum number of tokens per block, defaults to 512.
    :param int buffer_size: Buffer size for streaming, defaults to 10000.
    :param int seed: Random seed for reproducibility, defaults to 42.
    :param int min_text_length: Minimum text length to consider, defaults to 200.
    :param str text_field: Field containing text in the dataset, defaults to "text".
    :param int skip_samples: Number of samples to skip, defaults to 0.
    :param int max_samples: Maximum number of samples to use, defaults to None.
    :param Callable collator: Data collator callable, defaults to None.
    :param int num_workers: Number of worker processes, defaults to 4.
    :return DataLoader: PyTorch DataLoader for the dataset.

    Note: ``skip_samples`` and ``max_samples`` apply per worker when ``num_workers`` > 0.
    """
    dataset = HFStreamingDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        block_size=block_size,
        buffer_size=buffer_size,
        seed=seed,
        min_text_length=min_text_length,
        text_field=text_field,
        skip_samples=skip_samples,
        max_samples=max_samples,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


class DataCollatorForMaskedPermutedLanguageModeling:
    """Data collator for MPNet masked permuted language modeling."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pred_prob: float = 0.15,
        keep_prob: float = 0.10,
        rand_prob: float = 0.10,
        whole_word_mask: bool = True,
        use_fast: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the data collator.

        :param PreTrainedTokenizer tokenizer: Tokenizer for the model.
        :param float pred_prob: Probability that a token is predicted, defaults to 0.15.
        :param float keep_prob: Probability that predicted tokens are kept, defaults to 0.10.
        :param float rand_prob: Probability that predicted tokens are randomized, defaults to 0.10.
        :param bool whole_word_mask: Whether to use whole word masking, defaults to True.
        :param bool use_fast: Whether to use the fast span permutation, defaults to True.
        :param int random_seed: Random seed for reproducible permutations, defaults to None.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.pred_prob = pred_prob
        self.keep_prob = keep_prob
        self.rand_prob = rand_prob

        self.use_fast = use_fast
        if self.use_fast and make_span_perm is None:
            LOGGER.warning(
                "use_fast=True requested but fast span permutation extension is unavailable; "
                "falling back to Python implementation."
            )
            self.use_fast = False

        self.random_seed = random_seed
        self._np_rng = np.random.default_rng(random_seed) if random_seed is not None else None
        self._torch_generator = None
        if random_seed is not None:
            self._torch_generator = torch.Generator()
            self._torch_generator.manual_seed(random_seed)

        # Let's also create a byte tensor that maps words that begin with ##
        # We'll use this later on to do whole word masking
        if whole_word_mask:
            vocab_items = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
            # Require a reasonable number of ## tokens to avoid false positives for non-WordPiece.
            min_wordpiece_tokens = 100
            wordpiece_count = sum(1 for token, _ in vocab_items if token.startswith("##"))
            has_wordpiece = wordpiece_count >= min_wordpiece_tokens
            if not has_wordpiece:
                LOGGER.warning(
                    "whole_word_mask requested but tokenizer does not look WordPiece; "
                    "disabling whole-word masking."
                )
                self.whole_word_mask_map = None
            else:
                self.whole_word_mask_map = torch.ByteTensor(
                    [not token.startswith("##") for token, _ in vocab_items]
                )
        else:
            self.whole_word_mask_map = None

        # Finally, let's create a weight tensor that will make sure no special tokens are selected
        # when we are corrupting values later on
        weights = torch.ones(len(tokenizer.vocab), dtype=torch.float32)
        for idx in tokenizer.all_special_ids:
            weights[idx] = 0.0
        self.weights = weights / weights.sum()

    def __call__(self, examples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.collate_fn(examples)

    def get_rng_state(self) -> Optional[Dict[str, Any]]:
        """Return collator RNG state for deterministic resume.

        :return Optional[Dict[str, Any]]: RNG state payload or None.
        """
        state: Dict[str, Any] = {}
        if self._np_rng is not None:
            state["numpy"] = _serialize_np_generator_state(self._np_rng.bit_generator.state)
        if self._torch_generator is not None:
            state["torch"] = self._torch_generator.get_state()
        return state or None

    def set_rng_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore collator RNG state from a checkpoint.

        :param Optional[Dict[str, Any]] state: RNG state payload.
        :return None: This method returns nothing.
        """
        if not state:
            return
        if self._np_rng is not None and state.get("numpy") is not None:
            self._np_rng.bit_generator.state = _deserialize_np_generator_state(state["numpy"])
        if self._torch_generator is not None and state.get("torch") is not None:
            self._torch_generator.set_state(state["torch"])

    def collate_fn(self, examples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples for MPNet pretraining.

        :param Sequence[Dict[str, torch.Tensor]] examples: Tokenized examples.
        :return Dict[str, torch.Tensor]: Batch dictionary with masked inputs and targets.
        """

        # Start by creating a batch
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # Let's get the input IDs for the batch
        src_tokens = batch["input_ids"]

        # Get inline tokens so that we can keep track of the total non-padding tokens in the batch
        inline_tokens = src_tokens.view(-1)

        # Use these inline tokens with the padding_idx ignored to get the total number of tokens in
        # the batch
        ntokens = inline_tokens[inline_tokens != 1].numel()

        # Let's get the batch dimension
        sz = src_tokens.size()

        # Calculate the pred_size for this batch
        pred_size = round(sz[1] * self.pred_prob)

        # If the sequence is too short to have a masked token, we will mask the whole thing
        if pred_size == 0:
            pred_size = sz[1]

        # If we DO want to mask whole words, we use the span perm function which will permute whole
        # spans / words. Otherwise, we simply do a random permutation
        if self.whole_word_mask_map is not None:
            positions = torch.stack(
                [self.span_perm(src_tokens[i], pred_size) for i in range(sz[0])]
            )
        else:
            if self._torch_generator is not None:
                positions = torch.stack(
                    [torch.randperm(sz[1], generator=self._torch_generator) for i in range(sz[0])]
                )
            else:
                positions = torch.stack([torch.randperm(sz[1]) for i in range(sz[0])])

        # Now we actually do the permutation of the inputs based on the position outputs from above
        src_tokens, targets = self.permute_inputs(src_tokens, positions), None

        # Get the range of indices where mask tokens exist
        mask_range = range(sz[1] - pred_size, sz[1])

        # Extract targets, i.e. masked tokens, using the mask range
        targets = src_tokens[:, mask_range].contiguous()

        # Now mask the tokens using the mask_perm function. This will mask tokens and corrupt them
        # at a lower probability
        masked_tokens = self.mask_perm(targets.clone(), self.tokenizer.mask_token_id)

        # Now construct the postions and input IDs using the mask portion
        src_tokens = torch.cat((src_tokens, masked_tokens, masked_tokens), dim=1)
        positions = torch.cat(
            (positions, positions[:, mask_range], positions[:, mask_range]), dim=1
        )

        # Now load these up into collated form
        batch["targets"] = targets
        batch["input_ids"] = src_tokens
        batch["positions"] = positions
        batch["pred_size"] = targets.size(1)
        batch["ntokens"] = ntokens

        return batch

    # Let's define helper functions here
    def span_perm(self, x: torch.Tensor, pred_size: Optional[int] = None) -> torch.Tensor:
        """Generate a span permutation while preserving whole words.

        :param torch.Tensor x: Input IDs.
        :param int pred_size: Total number of tokens to predict, defaults to None.
        :return torch.Tensor: Permuted positions for the input sequence.
        """

        # Get a "mask" of which input IDs are the STARTS of words by using the map we created
        # earlier. This essentially uses the input IDs as indices to extract a 1 or 0 from the map
        word_begins_mask = self.whole_word_mask_map.gather(0, x)

        # Now let's get the positional indices of each word beginning. This will help us properly
        # create permutation "spans" later on. These are sorted from lowest to highest, so it might
        # look like:
        # [0, 1, 3, 4, 7, 10]
        # where each index represents the start of a word
        word_begins_idx = word_begins_mask.nonzero().view(-1).tolist()

        # Get the size of the positional sequence
        sz = len(word_begins_idx)

        # Create a permutation based on this size. In the example above, we would permute length
        # 6, thus would have something like:
        # np.array([1 5 3 2 4 0])
        # This will serve as the permuted indices of the positions stored in `word_begins_idx`
        if self._np_rng is not None:
            perm = self._np_rng.permutation(sz)
        else:
            perm = np.random.permutation(sz)

        # We also need to append the total length of the input sequence `x` for a reason that will
        # become clear below. Essentially, you may think about each of the values in
        # word_begins_mask as the start of the "span" while the next index is the end of the "span".
        # In the case of the last span (i.e. 10 in our example above), there would not be an end to
        # the span without appending the length of the input sequence
        word_begins_idx.append(x.size(0))

        # The best practice here will be to use the Cython implementation of the function, but we
        # leave the Python implementation as a branch here so that the reader can fully understand
        # how the permutations are generated without having to read into less friendly Cython
        if self.use_fast:
            # Pass the necessary components into the Cython function and get the ndarray back
            spans = make_span_perm(perm, word_begins_idx, x.size(0))

        else:
            # Finally, we use everything we've just created to actually create the spans below

            # Begin by defining a numpy array of the appropriate length (i.e. the size of the
            # source)
            spans = np.zeros(x.size(0), dtype=np.int64)

            # We create an index tracker called `g` that will only update when we are adding spans
            # to `spans`
            g = 0

            # Now iterate through each of the indices of `word_begins_idx`. Use those indices to
            # extract the permuted indices and use that to build the start and end span.
            #
            # Using the above example for just the first iteration:
            # i = 0
            # perm[i] = 1
            # perm[i] + 1 = 2
            # start = word_begins_idx[1] = 1
            # end = word_begins_idx[2] = 3
            #
            # Now that we have our start and end indices, we create our "span" by iterating over
            # those indices and using our `g` counter to add the span intos the final `spans` holder
            for i in range(len(word_begins_idx) - 1):
                start = word_begins_idx[perm[i]]
                end = word_begins_idx[perm[i] + 1]

                for j in range(start, end):
                    spans[g] = j
                    g += 1

        # Now we do one last shuffle of the masked indices to make sure they are also permuted
        if pred_size is not None:
            if self._np_rng is not None:
                self._np_rng.shuffle(spans[-pred_size:])
            else:
                np.random.shuffle(spans[-pred_size:])

        return torch.from_numpy(spans)

    def mask_perm(self, tokens: torch.Tensor, mask_idx: int = None) -> torch.Tensor:
        """
        Masks and corrupts tokens within the predicted portion of the input tokens (i.e. the last n
        tokens, where n = pred_size)

        Args:
            tokens: the tokens in `mask_range` which are the last n tokens where n = pred_size
            mask_idx: the index of the mask token, extracted from the tokenizer

        Returns:
            Returns the tokens with either masked or corrupted indices (or unchanged ones)
        """

        # Extract the probability that a token will be masked. This is what's left over after
        # subtracting the keep_prob and the rand_prob from 1. At default this means mask_prob is
        # 0.80
        mask_prob = 1.0 - self.rand_prob - self.keep_prob

        # Get the corruption probability, which is a function of how rand_prob and keep_prob play
        # with each other
        # You may think of this as the ratio of rand_prob to keep_prob
        # This is the probability that will be used to determine whether or not a token should be
        # randomly corrupted
        corrupt_prob = self.rand_prob / (
            1.0 - mask_prob
        )  # i.e. rand_prob / (rand_prob + keep_prob)

        # Now use torch's builtin bernoulli function to choose tokens (from a bernoulli dist.) that
        # will be masked and save their indices
        # More specifically, a tensor the size of `tokens` is created where each value is mask_prob
        # This is then passed to the bernoulli distribution, which either generates a 0 or 1 based
        # on that mask_prob
        # Finally we convert it to a boolean tensor for reasons that will be clear in the next step
        if self._torch_generator is not None:
            mask_indices = torch.rand(tokens.shape, generator=self._torch_generator) < mask_prob
        else:
            mask_indices = torch.rand(tokens.shape) < mask_prob

        # Now we get the indices where we want to corrupt the tokens using a similar approach to the
        # above logic. Using corrupt_prob, we use the bernoulli distribution to get a list of
        # indices that we will want to corrupt. We also test against the boolean tensor we made
        # above to make sure that we aren't corrupting any tokens that were already slated for
        # maskin
        if self._torch_generator is not None:
            corrupt_draw = torch.rand(tokens.shape, generator=self._torch_generator)
        else:
            corrupt_draw = torch.rand(tokens.shape)
        corrupt_indices = (corrupt_draw < corrupt_prob) & ~mask_indices

        # Now we mask and corrupt the tokens dictated by the indices above
        # We use the generate_random_tensor helper function to select random indices from the vocab
        tokens[mask_indices] = mask_idx
        tokens[corrupt_indices] = self.generate_random_tensor(corrupt_indices.sum().tolist()).to(
            tokens.device
        )

        return tokens

    def permute_inputs(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Permute input IDs using the provided positions.

        :param torch.Tensor inputs: Input IDs.
        :param torch.Tensor positions: Permuted positions.
        :return torch.Tensor: Permuted input IDs.
        """

        # Get the shape of the inputs, i.e., (batch_size, seq_len)
        sz = inputs.size()

        # Next we get an offset measurement. This is to calculate position permutations if there
        # are multiple source tokens in the batch. The arange function accepts a starting value, in
        # this case 0, and ending value, in this case sz[0] * sz[1] (the total number of values in
        # the batch), and a step size, in this case sz[1], which is the size of each input ID
        #
        # This means that each subsequent set of src_tokens in the batch will be offset by the
        # sequence length. This ensures we can do the permutation calculation properly below
        offset = torch.arange(0, sz[0] * sz[1], sz[1])

        # Now use this offset to amend the position values
        index = positions + offset.unsqueeze_(1)

        return inputs.reshape(-1)[index]

    def generate_random_tensor(self, sz: int) -> torch.Tensor:
        """Generate random token IDs for corruption.

        :param int sz: Number of random tokens to generate.
        :return torch.Tensor: Random token IDs tensor.
        """
        if sz <= 0:
            return torch.empty(0, dtype=torch.long)
        generator = self._torch_generator if self._torch_generator is not None else None
        return torch.multinomial(self.weights, sz, replacement=True, generator=generator)


class RandomSamplerWithSeed(Sampler[int]):
    """
    Random sampler based on the base Sampler class that allows for seeded random sampling such that
    epochs are reproducible. If a seed isn't provided, training will be truly random.
    """

    def __init__(self, data_source: Sized, epoch: int, random_seed: Optional[int] = None) -> None:
        """Initialize the sampler with an optional seed.

        :param Sized data_source: Data source to sample from.
        :param int epoch: Current epoch number.
        :param int random_seed: Optional random seed for reproducibility, defaults to None.
        """
        self.data_source = data_source
        self.epoch = epoch

        if random_seed is None:
            self.random_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            self.random_seed = random_seed

    def __iter__(self) -> Iterator[int]:
        with utils.numpy_seed(self.epoch + self.random_seed):
            shuffle = np.random.permutation(len(self.data_source))

        return iter(shuffle)

    def __len__(self) -> int:
        return len(self.data_source)
