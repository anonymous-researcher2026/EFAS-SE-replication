# ============================
# Aspect Classification (BERT + [CLS] + span-attention over entity tokens) — full end-to-end code
# Entity representation = attention-pooling over tokens between [A] and [/A]
# ============================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32 #16
LR = 3e-5 #3e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 10
N_SPLITS = 5
SEED = 1 #1

DATASET_PATH = "../data/dataset.csv"
OUT_DIR = "../results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Load tokenizer/model + add [A] tags
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {"additional_special_tokens": ["[A]", "[/A]"]}
tokenizer.add_special_tokens(special_tokens)

A_ID = tokenizer.convert_tokens_to_ids("[A]")
END_A_ID = tokenizer.convert_tokens_to_ids("[/A]")

# -----------------------------
# Model: [CLS] + span-attention(entity) -> FC -> logits
# -----------------------------
class BERT_Arch_SpanAttn(nn.Module):
    """
    Aspect classifier using:
      - c: [CLS] embedding
      - e: entity embedding = attention-pooling over tokens between [A] and [/A]
      - concat([c, e]) -> FC -> ReLU -> Dropout -> FC -> logits
    """
    def __init__(self, bert, hidden=768, dropout_p=0.3):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_p)

        # Attention scorer over token representations (entity span)
        # score_t = w^T h_t  (learned)
        self.entity_scorer = nn.Linear(hidden, 1, bias=False)

        self.fc1 = nn.Linear(hidden * 2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def _span_attention(self, H, input_ids, attention_mask):
        """
        H: (B, T, 768) last hidden states
        Build mask for entity span tokens strictly between [A] and [/A],
        then compute attention weights over that span and pool.
        """
        B, T, _ = H.shape

        # Find [A] and [/A] positions.
        # argmax returns 0 if not found; we guard with has_A/has_end.
        A_pos = (input_ids == A_ID)
        END_pos = (input_ids == END_A_ID)

        a_idx = torch.argmax(A_pos.int(), dim=1)        # (B,)
        end_idx = torch.argmax(END_pos.int(), dim=1)    # (B,)
        has_A = A_pos.any(dim=1)                        # (B,)
        has_END = END_pos.any(dim=1)                    # (B,)

        # Build token indices grid
        idx = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        # Entity span mask: tokens strictly inside (a_idx, end_idx)
        # also require valid attention_mask
        span_mask = (
            (idx > a_idx.unsqueeze(1)) &
            (idx < end_idx.unsqueeze(1)) &
            attention_mask.bool()
        )

        # If tags missing or end before start, span becomes empty; handle via has_span
        valid_tags = has_A & has_END & (end_idx > a_idx + 1)
        has_span = valid_tags & (span_mask.sum(dim=1) > 0)

        # Compute attention scores for all tokens, then mask out non-span
        scores = self.entity_scorer(H).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~span_mask, -1e9)

        # Softmax over time dimension
        weights = torch.softmax(scores, dim=1)      # (B, T)

        # Weighted sum
        e = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # (B, 768)

        # For rows with no valid span, set entity vector to zeros (safe fallback)
        if (~has_span).any():
            e = e.clone()
            e[~has_span] = 0.0

        return e  # (B, 768)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        H = out.last_hidden_state  # (B, T, 768)

        c = H[:, 0, :]  # [CLS] (B, 768)
        e = self._span_attention(H, input_ids, attention_mask)  # (B, 768)

        h = torch.cat([c, e], dim=1)  # (B, 1536)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)          # (B, 2)
        return logits


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def encode_texts(texts, max_len=MAX_LEN):
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids, attn_mask, labels = [t.to(DEVICE) for t in batch]
        optimizer.zero_grad()

        logits = model(input_ids, attn_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def eval_and_save(model, loader, out_csv_path):
    model.eval()
    y_true, y_pred = [], []
    decoded_sentences = []

    for batch in loader:
        input_ids, attn_mask, labels = [t.to(DEVICE) for t in batch]
        logits = model(input_ids, attn_mask)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        decoded_sentences.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    df_pred = pd.DataFrame({
        "Sentence": decoded_sentences,
        "Actual": y_true,
        "Predicted": y_pred
    })
    ensure_dir(os.path.dirname(out_csv_path))
    df_pred.to_csv(out_csv_path, index=False)

    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_str = classification_report(y_true, y_pred, digits=4)
    return report_dict, report_str


def flatten_report(report_dict, prefix=""):
    row = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                row[f"{prefix}{k}_{mk}"] = mv
        else:
            row[f"{prefix}{k}"] = v
    return row


# -----------------------------
# Main 5-fold CV
# -----------------------------
def run_cv():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(DATASET_PATH)
    if "Sentences_withTaggedA" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns: 'Sentences_withTaggedA' and 'label' in dataset")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        fold_dir = os.path.join(OUT_DIR, f"fold{fold}")
        ensure_dir(fold_dir)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Save training sentences for this fold
        train_sent_path = os.path.join(fold_dir, f"training_sentences_fold{fold}.csv")
        df_train[["Sentences_withTaggedA"]].to_csv(train_sent_path, index=False)

        # Encode
        train_ids, train_mask = encode_texts(df_train["Sentences_withTaggedA"].tolist(), max_len=MAX_LEN)
        test_ids, test_mask = encode_texts(df_test["Sentences_withTaggedA"].tolist(), max_len=MAX_LEN)

        y_train = torch.tensor(df_train["label"].tolist(), dtype=torch.long)
        y_test = torch.tensor(df_test["label"].tolist(), dtype=torch.long)

        train_ds = TensorDataset(train_ids, train_mask, y_train)
        test_ds = TensorDataset(test_ids, test_mask, y_test)

        train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=BATCH_SIZE)

        # Fresh BERT per fold (important)
        fold_bert = BertModel.from_pretrained(MODEL_NAME)
        fold_bert.resize_token_embeddings(len(tokenizer))
        model = BERT_Arch_SpanAttn(fold_bert).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)

            pred_path = os.path.join(fold_dir, f"ep{epoch}", f"predictions_fold{fold}.csv")
            report_dict, report_str = eval_and_save(model, test_loader, pred_path)

            print(f"\nFold {fold} | Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print("Test classification report:")
            print(report_str)
            print(f"Predictions saved to: {pred_path}")

            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            row.update(flatten_report(report_dict, prefix="test_"))
            all_rows.append(row)

        # Save model per fold
        torch.save(model.state_dict(), os.path.join(fold_dir, f"model_fold{fold}.pt"))

    # Save full summary across folds/epochs
    summary_path = os.path.join(OUT_DIR, "summary_all_folds_epochs.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # Average across folds by epoch for common metrics
    summary_df = pd.DataFrame(all_rows)
    agg_cols = [c for c in ["test_accuracy", "test_macro avg_f1-score", "test_weighted avg_f1-score"] if c in summary_df.columns]
    if agg_cols:
        agg = summary_df.groupby("epoch")[agg_cols].mean().reset_index()
        agg_path = os.path.join(OUT_DIR, "summary_avg_over_folds_by_epoch.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Saved epoch-level averaged summary to: {agg_path}")
    else:
        print("Note: aggregate metric columns not found; check sklearn report keys.")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_cv()
