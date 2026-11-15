"""Data, tokenization and decoding helpers for the UA location extraction task."""
import json
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

Span = Tuple[int, int]

ENTITY_COLUMNS = {"LOC": "loc_markers", "ORG": "org_markers", "PER": "per_markers"}
TOKEN_RE = re.compile(r"\w+(?:[-'ʼ’]\w+)*|[^\w\s]|\n")
NEWLINE = "<nl>"
EDGE_CHARS = " \t\n.,:;!?«»\"'()[]{}<>|/\\—–-“”„"
STREET_RE = re.compile(r"(?:вул|ул|просп|пр-т|пров|пер|бульв|б-р|пл|площ|шосе|узвіз|наб)[\w.]*\s", re.IGNORECASE)
HOUSE_RE = re.compile(r",?\s?\d+(?:[-/]?\w)?\b")
PAIRS = {"«": "»", "(": ")", "“": "”", "„": "“", '"': '"'}


# ---------------------------------------------------------------- metric (as published on Kaggle)

def hard_processing(text):
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def comp_metric(y_true: List[List[str]], y_pred: List[List[str]]):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p,  = 0.0, 0.0, 0.0, 0.0

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([hard_processing(s) for s in y_true_sample])
        y_pred_sample = set([hard_processing(s) for s in y_pred_sample])

        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)

    if tp + fp == 0:
        if p == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp/(tp + fp)

    if tp + fn == 0:
        if p == 0:
            recall = 1.0
        else:
            recall = 0.0
    else:
        recall = tp/(tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 8)


def bootstrap_ci(y_true, y_pred, n=2000, seed=67):
    """95% bootstrap interval of comp_metric, resampling whole texts."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(comp_metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    return tuple(np.percentile(scores, [2.5, 97.5]))


# ---------------------------------------------------------------- spans

def parse_markers(value: str) -> List[Span]:
    """'[(0, 16), (20, 25)]' -> [(0, 16), (20, 25)]"""
    return [tuple(s) for s in json.loads(value.replace("(", "[").replace(")", "]"))]


def merge_touching(spans: Sequence[Span]) -> List[Span]:
    """Joins spans with no gap between them: 'Івано' + '-' + 'Франківську' -> 'Івано-Франківську'."""
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def load_entities(df: pd.DataFrame, types: Sequence[str] = ("LOC", "ORG", "PER")) -> List[Dict[str, List[Span]]]:
    """Per text: {type: merged spans}, read from the *_markers columns."""
    columns = {t: df[ENTITY_COLUMNS[t]].map(parse_markers) for t in types}
    return [{t: merge_touching(columns[t].iat[i]) for t in types} for i in range(len(df))]


def label_names(types: Sequence[str]) -> List[str]:
    return ["O"] + [f"{prefix}-{t}" for t in types for prefix in "BI"]


def spans_to_tags(offsets: Sequence[Span], entities: Dict[str, List[Span]], text_len: int) -> List[str]:
    """BIO tag for every token. A token belongs to an entity if any of its characters do.
    LOC is written last, so it wins where a LOC is nested in an ORG."""
    owner = np.zeros(text_len, dtype=np.int32)
    kinds = [None]
    for t in sorted(entities, key=lambda t: t == "LOC"):
        for a, b in entities[t]:
            kinds.append(t)
            owner[a:b] = len(kinds) - 1
    tags, prev = [], 0
    for a, b in offsets:
        ids = owner[a:b]
        ids = ids[ids > 0]
        if len(ids) == 0:
            tags.append("O")
            prev = 0
            continue
        tags.append(f"{'I' if ids[0] == prev else 'B'}-{kinds[ids[0]]}")
        prev = ids[-1]  # a token can cover two touching spans, the next token continues the last one
    return tags


def make_posts(texts: Sequence[str], entities: List[Dict[str, List[Span]]], max_sentences=6,
               seps=(" ", "\n", "\n\n"), seed=67) -> Tuple[List[str], List[Dict[str, List[Span]]]]:
    """Glues consecutive sentences into pseudo-posts of 1..max_sentences with random separators and shifts
    the spans, so training inputs look like multiline Telegram posts and not single news sentences."""
    rng = np.random.default_rng(seed)
    posts, post_entities, i = [], [], 0
    while i < len(texts):
        k = int(rng.integers(1, max_sentences + 1))
        text, ents = "", {}
        for sentence, sentence_ents in zip(texts[i:i + k], entities[i:i + k]):
            if text:
                text += str(rng.choice(seps))
            for t, spans in sentence_ents.items():
                ents.setdefault(t, []).extend((a + len(text), b + len(text)) for a, b in spans)
            text += sentence
        posts.append(text)
        post_entities.append(ents)
        i += k
    return posts, post_entities


# ---------------------------------------------------------------- tokenizers

class WordTokenizer:
    """Regex words (hyphenated and apostrophe words kept whole), single punctuation chars and newlines."""
    specials = ["<pad>", "<unk>", NEWLINE]
    pad_id, unk_id = 0, 1

    def __init__(self, vocab: List[str]):
        self.itos = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}

    @classmethod
    def train(cls, texts, max_size=100_000, min_freq=2):
        counter = Counter(m.group() for text in texts for m in TOKEN_RE.finditer(text) if m.group() != "\n")
        words = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        return cls(cls.specials + words)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text) -> Tuple[List[int], List[Span]]:
        matches = list(TOKEN_RE.finditer(text))
        ids = [self.stoi.get(NEWLINE if m.group() == "\n" else m.group(), self.unk_id) for m in matches]
        return ids, [m.span() for m in matches]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "word", "vocab": self.itos}, f, ensure_ascii=False)


class BPETokenizer:
    """BPE trained on the same regex words, so subwords never cross a word boundary."""
    specials = WordTokenizer.specials
    pad_id, unk_id = 0, 1

    def __init__(self, tokenizer):
        self.tok = tokenizer

    @classmethod
    def train(cls, texts, vocab_size=32_000, min_frequency=2):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                      special_tokens=cls.specials, limit_alphabet=500, show_progress=False)
        words = (" ".join(m.group() for m in TOKEN_RE.finditer(text) if m.group() != "\n") for text in texts)
        tok.train_from_iterator(words, trainer=trainer)
        return cls(tok)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, text) -> Tuple[List[int], List[Span]]:
        matches = list(TOKEN_RE.finditer(text))
        if not matches:
            return [], []
        words = [NEWLINE if m.group() == "\n" else m.group() for m in matches]
        enc = self.tok.encode(words, is_pretokenized=True)
        offsets = [(matches[w].start() + a, matches[w].start() + b) if words[w] != NEWLINE else matches[w].span()
                   for w, (a, b) in zip(enc.word_ids, enc.offsets)]
        return enc.ids, offsets

    def save(self, path):
        self.tok.save(str(path))


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "word":
        return WordTokenizer(data["vocab"])
    from tokenizers import Tokenizer
    return BPETokenizer(Tokenizer.from_file(str(path)))


def load_vectors(path, vocab: List[str], pad_id=0, seed=67) -> Tuple[torch.Tensor, np.ndarray]:
    """Embedding matrix for `vocab` from a fastText/GloVe text file (.vec or .vec.gz, 'word v1 v2 ...').
    A word missing in the file takes the vector of its lowercase form; the rest are random with the std of
    the found vectors. Returns the matrix and a mask of the rows that came from the file."""
    import gzip
    wanted = set(vocab) | {w.lower() for w in vocab}
    found = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word, _, rest = line.rstrip().partition(" ")
            if word in wanted and rest.count(" ") > 0:  # the fastText header line has two numbers only
                found[word] = np.asarray(rest.split(" "), dtype=np.float32)
    dim = len(next(iter(found.values())))
    std = float(np.std(np.stack(list(found.values())[:10_000])))
    matrix = np.random.default_rng(seed).normal(0, std, (len(vocab), dim)).astype(np.float32)
    mask = np.zeros(len(vocab), dtype=bool)
    for i, w in enumerate(vocab):
        vector = found.get(w, found.get(w.lower()))
        if vector is not None:
            matrix[i], mask[i] = vector, True
    matrix[pad_id] = 0
    return torch.from_numpy(matrix), mask


# ---------------------------------------------------------------- dataset

class NERDataset(Dataset):
    """Tokenizes on the fly. `entities` (from load_entities) is optional - without it there are no labels."""

    def __init__(self, texts: Sequence[str], tokenizer, types: Sequence[str] = ("LOC",),
                 entities: Optional[List[Dict[str, List[Span]]]] = None, max_tokens: Optional[int] = None,
                 targets: Optional[Sequence[float]] = None):
        self.texts = list(texts)
        self.targets = targets  # one number per text, e.g. is_test for adversarial validation
        self.tokenizer = tokenizer
        self.entities = entities
        self.types = list(types)
        self.label2id = {name: i for i, name in enumerate(label_names(types))}
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        ids, offsets = self.tokenizer.encode(text)
        if not ids:  # nothing to read, keep one token so packing works
            ids, offsets = [self.tokenizer.unk_id], [(0, 0)]
        ids, offsets = ids[:self.max_tokens], offsets[:self.max_tokens]
        item = {"input_ids": ids, "offsets": offsets, "index": i}
        if self.targets is not None:
            item["target"] = float(self.targets[i])
        if self.entities is not None:
            ents = {t: self.entities[i].get(t, []) for t in self.types}
            item["labels"] = [self.label2id[tag] for tag in spans_to_tags(offsets, ents, len(text))]
        return item


class Collator:
    """Pads input_ids with pad_id and labels with -100; returns true lengths for pack_padded_sequence."""

    def __init__(self, pad_id=0, label_pad_id=-100):
        self.pad_id, self.label_pad_id = pad_id, label_pad_id

    def __call__(self, batch):
        lengths = torch.tensor([len(b["input_ids"]) for b in batch])
        input_ids = torch.full((len(batch), int(lengths.max())), self.pad_id, dtype=torch.long)
        for row, b in enumerate(batch):
            input_ids[row, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        out = {"input_ids": input_ids, "lengths": lengths, "index": torch.tensor([b["index"] for b in batch])}
        if "target" in batch[0]:
            out["target"] = torch.tensor([b["target"] for b in batch])
        if "labels" in batch[0]:
            labels = torch.full_like(input_ids, self.label_pad_id)
            for row, b in enumerate(batch):
                labels[row, :len(b["labels"])] = torch.tensor(b["labels"])
            out["labels"] = labels
        return out


# ---------------------------------------------------------------- decoding

def _is_word_char(text, i):
    return 0 <= i < len(text) and (text[i].isalnum() or text[i] == "_")


def snap_to_word(text, a, b) -> Span:
    """Extends a span to whole words, including hyphens/apostrophes inside a word."""
    while a > 0 and (_is_word_char(text, a - 1) or (text[a - 1] in "-'ʼ’" and _is_word_char(text, a - 2))):
        a -= 1
    while b < len(text) and (_is_word_char(text, b) or (text[b] in "-'ʼ’" and _is_word_char(text, b + 1))):
        b += 1
    return a, b


def tags_to_spans(loc: np.ndarray, begin: np.ndarray, offsets: Sequence[Span], scheme="lenient") -> List[Span]:
    """Token flags -> character spans. `loc[t]` - token t is a location, `begin[t]` - it starts one.
    lenient: an I- token after O starts a new entity (seqeval default); strict: it is dropped;
    io: B/I is ignored, every run of location tokens is one entity."""
    spans, current = [], None
    for t, (a, b) in enumerate(offsets):
        if not loc[t]:
            current = None
            continue
        if current is None and not begin[t] and scheme == "strict":
            continue
        if current is None or (begin[t] and scheme != "io"):
            spans.append([a, b])
            current = spans[-1]
        else:
            current[1] = b
    return [tuple(s) for s in spans]


def clean_spans(text, spans: Sequence[Span], snap=True, merge_hyphens=True, bridge_dots=False,
                house_numbers=False, strip_symbols=False) -> List[str]:
    """Spans -> unique location strings cut from the text.
    merge_hyphens: 'Івано' + '-' + 'Франківськ' is one entity; bridge_dots: so is 'вул' + '. ' + 'Шевченка';
    house_numbers: a span that starts with a street word takes the number after it ('вул. Шевченка' + ', 12');
    strip_symbols: emoji and other non-letter characters are cut from both ends ('Києві🏗' -> 'Києві')."""
    gaps = ({"-", "–"} if merge_hyphens else set()) | ({".", ". "} if bridge_dots else set())
    if snap:
        spans = [snap_to_word(text, a, b) for a, b in spans]
    spans = sorted(spans)
    merged = []
    for a, b in spans:
        gap = text[merged[-1][1]:a] if merged else None
        if merged and (a <= merged[-1][1] or gap in gaps):
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        s = text[a:b].strip(EDGE_CHARS)
        if strip_symbols:
            s = re.sub(r"^\W+|\W+$", "", s)
        b = a + text[a:b].find(s) + len(s)
        house = HOUSE_RE.match(text, b) if house_numbers and STREET_RE.match(s) else None
        if house:
            s, b = text[b - len(s):house.end()], house.end()
        closer = text[b:b + 1]
        if any(close == closer and (s.count(op) % 2 if op == close else s.count(op) > s.count(close))
               for op, close in PAIRS.items()):
            s += closer  # 'ТРЦ «Гулівер' -> 'ТРЦ «Гулівер»'
        if hard_processing(s) and s not in out:
            out.append(s)
    return out


def decode_locations(text, offsets, probs: np.ndarray, labels: List[str], threshold: Optional[float] = None,
                     scheme="lenient", **clean_kwargs) -> List[str]:
    """probs [T, C] -> location strings. threshold=None takes the argmax tag; otherwise a token is a location
    if P(B-LOC) + P(I-LOC) >= threshold."""
    b, i = labels.index("B-LOC"), labels.index("I-LOC")
    probs = probs[:len(offsets)]
    if threshold is None:
        tags = probs.argmax(-1)
        loc, begin = (tags == b) | (tags == i), tags == b
    else:
        loc, begin = probs[:, b] + probs[:, i] >= threshold, probs[:, b] >= probs[:, i]
    return clean_spans(text, tags_to_spans(loc, begin, offsets, scheme), **clean_kwargs)


def predict_locations(texts, offsets, probs, labels, **decode_kwargs) -> List[List[str]]:
    """decode_locations for a list of texts. `offsets[i]`, `probs[i]` belong to `texts[i]`."""
    return [decode_locations(t, o, p, labels, **decode_kwargs) for t, o, p in zip(texts, offsets, probs)]


def search_threshold(texts, offsets, probs, labels, gold, thresholds=np.arange(0.1, 0.91, 0.05), **decode_kwargs):
    """Competition metric for the argmax decoding and for every threshold on P(B-LOC) + P(I-LOC)."""
    scores = {"argmax": comp_metric(gold, predict_locations(texts, offsets, probs, labels, **decode_kwargs))}
    for thr in thresholds:
        pred = predict_locations(texts, offsets, probs, labels, threshold=thr, **decode_kwargs)
        scores[round(float(thr), 2)] = comp_metric(gold, pred)
    return pd.Series(scores, name="F1")


def word_probs(text, offsets, probs: np.ndarray) -> Tuple[List[Span], np.ndarray]:
    """Token probabilities -> one row per regex word (its first subword), so models with different
    tokenizers get the same shape and can be averaged. Words cut off by max_tokens get the O class."""
    words = word_spans(text)
    first = {}
    for t, (a, _) in enumerate(offsets[:len(probs)]):
        first.setdefault(a, t)
    out = np.zeros((len(words), probs.shape[1]), dtype=np.float32)
    out[:, 0] = 1.0
    for w, (a, _) in enumerate(words):
        if a in first:
            out[w] = probs[first[a]]
    return words, out


def word_spans(text) -> List[Span]:
    """Offsets of the regex words - the rows of word_probs."""
    return [m.span() for m in TOKEN_RE.finditer(text)]


def loc_probs(probs: np.ndarray, labels: List[str]) -> np.ndarray:
    """[T, C] over any label set -> [T, 3] over O, B-LOC, I-LOC (ORG and PER go to O)."""
    b, i = probs[:, labels.index("B-LOC")], probs[:, labels.index("I-LOC")]
    return np.stack([1 - b - i, b, i], axis=1)


def save_probs(path, probs: Dict[str, List[np.ndarray]]):
    """{set name: one [W, C] array per text} -> one .npz file."""
    arrays = {}
    for name, rows in probs.items():
        arrays[name] = np.concatenate(rows).astype(np.float16)
        arrays[name + "_len"] = np.array([len(r) for r in rows])
    np.savez_compressed(path, **arrays)


def load_probs(path) -> Dict[str, List[np.ndarray]]:
    data = np.load(path)
    return {name: np.split(data[name].astype(np.float32), np.cumsum(data[name + "_len"])[:-1])
            for name in data.files if not name.endswith("_len")}


def ensemble_probs(model_probs: Sequence[Sequence[np.ndarray]], weights: Optional[Sequence[float]] = None):
    """Weighted mean of the word-level probabilities of several models: model_probs[m][i] is [W_i, C]."""
    weights = np.ones(len(model_probs)) if weights is None else np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return [sum(w * p[i] for w, p in zip(weights, model_probs)) for i in range(len(model_probs[0]))]


def gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """L2 norm of the gradient of every parameter tensor - call after backward (on_after_backward)."""
    return {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}


def plot_gradient_flow(history: pd.DataFrame, path=None):
    """history: rows - logged steps, columns - parameter names (from gradient_norms).
    Plots every tensor's share of the total gradient norm. Shares, not raw norms: with 16-bit mixed precision
    on_after_backward sees gradients of the loss multiplied by the GradScaler scale (65536, doubled every
    2000 steps), so the absolute values are not gradient norms, but the ratios between layers are."""
    import matplotlib.pyplot as plt
    weights = history[[c for c in history.columns if "bias" not in c]]
    weights = weights.rename(columns=lambda c: c.replace("encoder.", "").replace("rnn.", "").replace(".weight", ""))
    shares = weights.div(np.sqrt((weights ** 2).sum(axis=1)), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    shares.plot(ax=axes[0], logy=True, lw=1, marker="o", ms=2)
    axes[0].set(title="Share of the total gradient norm during training", xlabel="step", ylabel="share (log)")
    axes[0].legend(fontsize=7, ncol=2)
    shares.mean().plot.bar(ax=axes[1], logy=True)
    axes[1].set(title="Mean share per layer", ylabel="share (log)")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def make_submission(test_df: pd.DataFrame, locations: List[List[str]], path):
    sub = pd.DataFrame({"text_id": test_df["text_id"],
                        "locations": [json.dumps(l, ensure_ascii=False) for l in locations]})
    sub.to_csv(path, index=False)
    return sub
