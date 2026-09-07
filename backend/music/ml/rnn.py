
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import math
import random
from .music_config import DT_VOCAB, VEL_VOCAB
from ..services import midi_parser as Parser
from ..services import midi_tester as midi_tester

from .base_model import BaseMusicModel, SEED_NOTES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQUENCE_LENGTH = 64
MODEL_DIR = "./music/trained_models/rnn"
MODEL_NUM = 1

from .music_config import (
    PITCH_CLASS_VOCAB, OCTAVE_VOCAB, PITCH_VOCAB, VEL_VOCAB, DT_VOCAB, DUR_VOCAB, SUS_VOCAB,
    MAX_PITCH, MAX_VELOCITY, MAX_DURATION,
)

os.makedirs(MODEL_DIR, exist_ok=True)

class MusicDataset(Dataset):

    def __init__(self, songs, seq_len, augment=True):
        self.songs = songs
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.songs) * 50

    def __getitem__(self, idx):
        song = self.songs[idx % len(self.songs)]

        # mix random + deterministic sampling - validation uses deterministic windows
        if self.augment and torch.rand(1).item() < 0.7:
            start = torch.randint(0, len(song) - self.seq_len, (1,)).item()
        else:
            start = (idx * self.seq_len) % (len(song) - self.seq_len)

        seq = song[start:start + self.seq_len]

        notes = [n[0] for n in seq]
        others = [(n[1], n[2], n[3]) for n in seq]
        times = [[n[4]] for n in seq]  # fraction of song elapsed: 0 (start) .. 1 (end)

        return notes, others, times


def collate_fn(batch):
    notes, others, times = zip(*batch)
    return (
        torch.tensor(notes, dtype=torch.long),
        torch.tensor(others, dtype=torch.long),
        torch.tensor(times, dtype=torch.float32)
    )


# =========================
# MODEL
# =========================
class MusicRNN(BaseMusicModel):

    def __init__(
        self,
        pitch_class_vocab=12,
        octave_vocab=11,
        vel_vocab=VEL_VOCAB,
        dt_vocab=DT_VOCAB,
        sus_vocab=SUS_VOCAB,
        hidden_size=512, # number of features (or dimensions) in the hidden state vector
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.pc_emb = nn.Embedding(pitch_class_vocab, 16)
        self.oct_emb = nn.Embedding(octave_vocab, 16)
        self.vel_emb = nn.Embedding(vel_vocab, 8)
        self.dt_emb = nn.Embedding(dt_vocab, 8)
        self.sus_emb = nn.Embedding(sus_vocab, 8)
        self.time_proj = nn.Linear(1, 8)  # continuous position-in-song: 0 (start) .. 1 (end)

        self.dropout = nn.Dropout(dropout)

        self.event_proj = nn.Linear(64, hidden_size)

        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.pc_head = nn.Linear(hidden_size, pitch_class_vocab)
        self.oct_head = nn.Linear(hidden_size, octave_vocab)
        self.vel_head = nn.Linear(hidden_size, vel_vocab)
        self.dt_head = nn.Linear(hidden_size, dt_vocab)
        self.sus_head = nn.Linear(hidden_size, sus_vocab)

    def forward(self, notes, others, times):

        pc = notes % 12
        octv = notes // 12

        vel = others[:, :, 0].long()
        dt  = others[:, :, 1].long()
        sus = others[:, :, 2].long()

        x = torch.cat([
            self.pc_emb(pc),
            self.oct_emb(octv),
            self.vel_emb(vel),
            self.dt_emb(dt),
            self.sus_emb(sus),
            self.time_proj(times)
        ], dim=-1)

        x = self.dropout(x)
        x = self.event_proj(x) # Projection into model space
        x = self.dropout(x)

        out, _ = self.rnn(x) # shape: (16, 64, 256) (batch, seq len, hidden size)
        out = self.dropout(out)

        return (
            self.pc_head(out),
            self.oct_head(out),
            self.vel_head(out),
            self.dt_head(out),
            self.sus_head(out)
        )

    def fineTune(self, song):
        fineTune(self, song)
        return self

    def generate(self, seedSong):
        return compose(self, seedSong)


def _rnn_losses(model, notes, others, times, ce):
    pc_logits, oct_logits, vel_logits, dt_logits, sus_logits = model(notes, others, times)

    pc = notes % 12
    octv = notes // 12

    loss_pc = ce(pc_logits[:, :-1].reshape(-1, 12), pc[:, 1:].reshape(-1))
    loss_oct = ce(oct_logits[:, :-1].reshape(-1, 11), octv[:, 1:].reshape(-1))

    loss_vel = ce(
        vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
        others[:, 1:, 0].reshape(-1)
    )

    loss_dt = ce(
        dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
        others[:, 1:, 1].reshape(-1)
    )

    loss_sus = ce(
        sus_logits[:, :-1].reshape(-1, sus_logits.size(-1)),
        others[:, 1:, 2].reshape(-1)
    )

    return 2 * loss_pc + loss_oct + loss_vel + loss_dt + loss_sus


@torch.no_grad()
def evaluate(model, loader, ce, use_amp):
    model.eval()
    total = 0.0
    count = 0

    for notes, others, times in loader:
        notes, others, times = notes.to(DEVICE), others.to(DEVICE), times.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = _rnn_losses(model, notes, others, times, ce)
        total += loss.item()
        count += 1

    model.train()
    return total / max(1, count)


def train(model, dataloader, val_loader=None, epochs=10, lr=1e-3, warmup_steps=200, checkpoint_every=1):

    use_amp = torch.cuda.is_available()
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr) # updates weights using gradients
    ce = nn.CrossEntropyLoss() # used because all outputs are classification problems
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    total_steps = epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # "latest" is overwritten every checkpoint so a preempted/interrupted run never
    # loses more than checkpoint_every epochs of progress; "best" tracks the lowest
    # validation loss seen so far, protecting against late overfitting on a long run
    latest_path = os.path.join(MODEL_DIR, f"pretrained_{MODEL_NUM}.pt")
    best_path = os.path.join(MODEL_DIR, f"pretrained_{MODEL_NUM}_best.pt")
    best_val_loss = float('inf')

    num_batches = len(dataloader)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for batch_idx, (notes, others, times) in enumerate(dataloader, start=1):
            notes, others, times = notes.to(DEVICE), others.to(DEVICE), times.to(DEVICE)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = _rnn_losses(model, notes, others, times, ce)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            total += loss.item()

            if batch_idx % 100 == 0 or batch_idx == num_batches:
                print(f"  rnn epoch {epoch} | batch {batch_idx}/{num_batches} | loss so far {total:.4f}", flush=True)

        msg = f"rnn epoch {epoch} | train loss {total:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, ce, use_amp)
            msg += f" | val loss {val_loss:.4f}"

        print(msg, flush=True)

        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), latest_path)
            print(f"  -> checkpoint saved to {latest_path}", flush=True)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best val loss {best_val_loss:.4f}, saved to {best_path}", flush=True)

def loadModel():
    # model_path = os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt")
    model_path = os.path.join(MODEL_DIR, f"pretrained_{MODEL_NUM}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained RNN model found at {model_path}")

    model = MusicRNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

def fineTune(model, song, seq_len=64, epochs=4, batch_size=16, lr=3e-5):

    if len(song) <= seq_len:
        raise ValueError(
            f"Seed song has {len(song)} notes, but fine-tuning needs more than "
            f"seq_len={seq_len} notes."
        )

    use_amp = torch.cuda.is_available()
    model = model.to(DEVICE)
    model.train()

    # 1) create dataset with ONLY this song
    dataset = MusicDataset([song], seq_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # safer for single-song fine-tune
        collate_fn=collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        total_loss = 0.0

        for notes, others, times in loader:
            notes, others, times = notes.to(DEVICE), others.to(DEVICE), times.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                pc_logits, oct_logits, vel_logits, dt_logits, sus_logits = model(notes, others, times)

                # split ground truth
                pc = notes % 12
                octv = notes // 12

                loss_pc = loss_fn(
                    pc_logits[:, :-1].reshape(-1, 12),
                    pc[:, 1:].reshape(-1)
                )

                loss_oct = loss_fn(
                    oct_logits[:, :-1].reshape(-1, 11),
                    octv[:, 1:].reshape(-1)
                )

                loss_vel = loss_fn(
                    vel_logits[:, :-1].reshape(-1, vel_logits.size(-1)),
                    others[:, 1:, 0].reshape(-1)
                )

                loss_dt = loss_fn(
                    dt_logits[:, :-1].reshape(-1, dt_logits.size(-1)),
                    others[:, 1:, 1].reshape(-1)
                )

                loss_sus = loss_fn(
                    sus_logits[:, :-1].reshape(-1, sus_logits.size(-1)),
                    others[:, 1:, 2].reshape(-1)
                )

                loss = 2 * loss_pc + loss_oct + loss_vel + loss_dt + loss_sus

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"[fine-tune] epoch {epoch+1} | loss {total_loss:.4f}")

    return model

@torch.no_grad() # do not compute gradients
def compose(model, seedSong):

    model = model.to(DEVICE)
    model.eval()

    target_notes = len(seedSong)
    seedSong = seedSong[:SEED_NOTES]

    seq_notes = [n[0] for n in seedSong]
    seq_others = [[n[1], n[2], n[3]] for n in seedSong]
    seq_times = [[min(1.0, i / target_notes)] for i in range(len(seedSong))]

    generated = []

    def sample(logits, temp=0.9):
        probs = torch.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1).item() # Turns raw model scores into probabilities

    while len(generated) < target_notes:

        n = torch.tensor([seq_notes], dtype=torch.long, device=DEVICE)
        o = torch.tensor([seq_others], dtype=torch.long, device=DEVICE)
        t = torch.tensor([seq_times], dtype=torch.float32, device=DEVICE)

        pc_logits, oct_logits, vel_logits, dt_logits, sus_logits = model(n, o, t) # predicted distributions for each time step

        pc = sample(pc_logits[:, -1])
        octv = sample(oct_logits[:, -1])
        vel = sample(vel_logits[:, -1])
        dt = sample(dt_logits[:, -1])
        sus = sample(sus_logits[:, -1])

        next_note = min(octv * 12 + pc, 127)

        generated.append((next_note, vel, dt, sus))

        time_value = min(1.0, len(generated) / target_notes)

        seq_notes.append(next_note) # update context
        seq_others.append([vel, dt, sus])
        seq_times.append([time_value])

        # keep context stable
        if len(seq_notes) > SEQUENCE_LENGTH:
            seq_notes = seq_notes[-SEQUENCE_LENGTH:]
            seq_others = seq_others[-SEQUENCE_LENGTH:]
            seq_times = seq_times[-SEQUENCE_LENGTH:]

    print("generated notes:", len(generated))

    return [(n[0], n[1], n[2], n[3]) for n in seedSong] + generated

def trainModel(songs, val_split=0.1):
    shuffled = songs[:]
    random.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - val_split))) if len(shuffled) > 1 else len(shuffled)
    train_songs, val_songs = shuffled[:split_idx], shuffled[split_idx:]

    dataset = MusicDataset(train_songs, SEQUENCE_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_songs:
        val_dataset = MusicDataset(val_songs, SEQUENCE_LENGTH, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    model = MusicRNN()

    train(model, loader, val_loader=val_loader, epochs=10)

    return model


# def composeMusic(seedSong):
#     parser = Parser.MidiParser(MAX_VELOCITY, MAX_TIME, MAX_DURATION)
#     # model = trainModel(songs)
#
#     model = loadModel()
#
#     model = fineTune(model, seedSong, epochs=2, lr=3e-5)
#
#     # seedSong = songs[0][:SEQUENCE_LENGTH]
#
#     generated = compose(model, seedSong)
#     # generatedNotes = parser.convertedNotes(generated)
#     # midi_tester.testMidi(generatedNotes, "midiRNN1.mid")
#
#     return generatedNotes
