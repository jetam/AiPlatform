
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ..services import midi_parser as Parser
from ..services import midi_tester as midi_tester
import os

from .base_model import BaseMusicModel, SEED_NOTES


MODEL_DIR = "./music/trained_models/transformer1"
MODEL_NUM = 1
MODEL_NUM = 1

os.makedirs(MODEL_DIR, exist_ok=True)

from .music_config import (
    PITCH_CLASS_VOCAB, OCTAVE_VOCAB, PITCH_VOCAB, VEL_VOCAB, DT_VOCAB, DUR_VOCAB, SUS_VOCAB,
    MAX_PITCH, MAX_VELOCITY, MAX_TIME, MAX_DURATION, DT_MAX_SECONDS, TARGET_SECONDS,
)

TOKEN_VOCAB = PITCH_VOCAB * VEL_VOCAB * DT_VOCAB

MAX_SEQ_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_token(pitch: int, vel: int, dt: int) -> int:
    return pitch * (VEL_VOCAB * DT_VOCAB) + vel * DT_VOCAB + dt


def decode_token(token: int):
    dt = token % DT_VOCAB
    vel = (token // DT_VOCAB) % VEL_VOCAB
    pitch = token // (VEL_VOCAB * DT_VOCAB)
    return pitch, vel, dt


class MusicDataset(Dataset):
    def __init__(self, songs, seq_len=256):
        self.seq_len = seq_len
        self.data = []

        for song in songs:
            tokens = [encode_token(n[0], n[1], n[2]) for n in song]
            pitches = [n[0] for n in song]
            sustains = [n[3] for n in song]
            times = [n[4] for n in song]
            for i in range(0, len(tokens) - seq_len - 1, seq_len):
                tok_chunk = tokens[i:i + seq_len + 1]
                pit_chunk = pitches[i:i + seq_len + 1]
                sus_chunk = sustains[i:i + seq_len + 1]
                time_chunk = times[i:i + seq_len + 1]
                if len(tok_chunk) == seq_len + 1:
                    self.data.append((tok_chunk, pit_chunk, sus_chunk, time_chunk))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tok_chunk, pit_chunk, sus_chunk, time_chunk = self.data[idx]

        # pitch transposition augmentation: shift ±6 semitones per chunk
        shift = random.randint(-6, 6)
        if shift != 0:
            pit_chunk = [max(0, min(127, p + shift)) for p in pit_chunk]
            tok_chunk = [encode_token(p, *decode_token(t)[1:]) for p, t in zip(pit_chunk, tok_chunk)]

        def make_rel(pitches):
            p = torch.tensor(pitches, dtype=torch.long)
            rel = torch.zeros_like(p)
            rel[1:] = p[1:] - p[:-1]
            return torch.clamp(rel + 64, 0, 127)

        x_tok = torch.tensor(tok_chunk[:-1], dtype=torch.long)
        x_rel = make_rel(pit_chunk[:-1])
        x_sus = torch.tensor(sus_chunk[:-1], dtype=torch.long)
        x_time = torch.tensor(time_chunk[:-1], dtype=torch.float32).unsqueeze(-1)

        y_data = [decode_token(t) for t in tok_chunk[1:]]
        y_pc = torch.tensor([p % 12      for p, v, d in y_data], dtype=torch.long)
        y_po = torch.tensor([p // 12     for p, v, d in y_data], dtype=torch.long)
        y_v  = torch.tensor([v           for p, v, d in y_data], dtype=torch.long)
        y_d  = torch.tensor([d           for p, v, d in y_data], dtype=torch.long)
        y_sus = torch.tensor(sus_chunk[1:], dtype=torch.long)

        return (x_tok, x_rel, x_sus, x_time), (y_pc, y_po, y_v, y_d, y_sus)


class RelativeAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = math.sqrt(self.d_k)

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        # one embedding per relative distance
        self.rel_pos_emb = nn.Embedding(MAX_SEQ_LEN, self.d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        B, T, _ = x.shape
        H, D = self.nhead, self.d_k

        Q = self.q(x).view(B, T, H, D).permute(0, 2, 1, 3)  # [B, H, T, D]
        K = self.k(x).view(B, T, H, D).permute(0, 2, 1, 3)
        V = self.v(x).view(B, T, H, D).permute(0, 2, 1, 3)

        content = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # dist[i, j] = i - j  (how far query i looks back to key j)
        idx = torch.arange(T, device=x.device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0, max=MAX_SEQ_LEN - 1)  # [T, T]
        R = self.rel_pos_emb(dist)                                     # [T, T, D]
        positional = torch.einsum('bhid,ijd->bhij', Q, R) / self.scale # [B, H, T, T]

        attn = (content + positional).masked_fill(causal_mask, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        return self.out(out)


class RelativeTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, ff_mult=4):
        super().__init__()
        self.attn  = RelativeAttention(d_model, nhead, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        x = x + self.drop(self.attn(self.norm1(x), causal_mask))
        x = x + self.ff(self.norm2(x))
        return x


class MusicTransformerT1(BaseMusicModel):
    COND_DIM = 32

    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()

        self.tok_emb       = nn.Embedding(TOKEN_VOCAB, d_model)
        self.rel_pitch_emb = nn.Embedding(128, d_model)  # note-to-note interval
        self.sus_emb       = nn.Embedding(SUS_VOCAB, d_model)  # sustain pedal state for this note
        self.time_proj     = nn.Linear(1, d_model)  # position in piece: 0 (start) .. 1 (end)

        self.layers = nn.ModuleList([
            RelativeTransformerLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # heads are chained via the chain rule so each field conditions on
        # the ones already decided for the same note: pc -> po -> v -> dt -> sus
        d_cond = self.COND_DIM
        self.pc_cond = nn.Embedding(PITCH_CLASS_VOCAB, d_cond)
        self.po_cond = nn.Embedding(OCTAVE_VOCAB, d_cond)
        self.v_cond  = nn.Embedding(VEL_VOCAB, d_cond)
        self.d_cond  = nn.Embedding(DT_VOCAB, d_cond)

        self.out_pc  = nn.Linear(d_model, PITCH_CLASS_VOCAB)
        self.out_po  = nn.Linear(d_model + d_cond,     OCTAVE_VOCAB)
        self.out_v   = nn.Linear(d_model + d_cond * 2, VEL_VOCAB)
        self.out_d   = nn.Linear(d_model + d_cond * 3, DT_VOCAB)
        self.out_sus = nn.Linear(d_model + d_cond * 4, SUS_VOCAB)

    def causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def encode(self, tokens, rel_pitch, sustain, time):
        x = self.tok_emb(tokens) + self.rel_pitch_emb(rel_pitch) + self.sus_emb(sustain) + self.time_proj(time)

        mask = self.causal_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

    def decode_heads(self, h, pc=None, po=None, v=None, dt=None):
        # p(pc,po,v,dt,sus|h) = p(pc|h) p(po|h,pc) p(v|h,pc,po) p(dt|h,pc,po,v) p(sus|h,pc,po,v,dt)
        # pc/po/v/dt are ground-truth tensors for teacher forcing; if omitted,
        # each head falls back to its own argmax. compose() does its own
        # top-p sampling between head calls instead of relying on that
        # fallback.
        pc_logits = self.out_pc(h)
        pc_in = pc if pc is not None else pc_logits.argmax(-1)
        h_po = torch.cat([h, self.pc_cond(pc_in)], dim=-1)

        po_logits = self.out_po(h_po)
        po_in = po if po is not None else po_logits.argmax(-1)
        h_v = torch.cat([h_po, self.po_cond(po_in)], dim=-1)

        v_logits = self.out_v(h_v)
        v_in = v if v is not None else v_logits.argmax(-1)
        h_d = torch.cat([h_v, self.v_cond(v_in)], dim=-1)

        d_logits = self.out_d(h_d)
        d_in = dt if dt is not None else d_logits.argmax(-1)
        h_sus = torch.cat([h_d, self.d_cond(d_in)], dim=-1)

        sus_logits = self.out_sus(h_sus)

        return pc_logits, po_logits, v_logits, d_logits, sus_logits

    def forward(self, tokens, rel_pitch, sustain, time, pc=None, po=None, v=None, dt=None):
        h = self.encode(tokens, rel_pitch, sustain, time)
        return self.decode_heads(h, pc=pc, po=po, v=v, dt=dt)

    def fineTune(self, song):
        fineTune(self, song)
        return self

    def generate(self, seedSong, targetSeconds=TARGET_SECONDS):
        return compose(self, seedSong, targetSeconds=targetSeconds)


def loss_fn(logits, targets):
    lpc, lpo, lv, ld, lsus = logits
    ypc, ypo, yv, yd, ysus = targets
    return (
        F.cross_entropy(lpc.reshape(-1, PITCH_CLASS_VOCAB), ypc.reshape(-1)) +
        F.cross_entropy(lpo.reshape(-1, OCTAVE_VOCAB),      ypo.reshape(-1)) +
        F.cross_entropy(lv.reshape(-1,  VEL_VOCAB),         yv.reshape(-1))  +
        F.cross_entropy(ld.reshape(-1,  DT_VOCAB),          yd.reshape(-1))  +
        F.cross_entropy(lsus.reshape(-1, SUS_VOCAB),        ysus.reshape(-1))
    )


def train(model, songs, epochs=5, batch_size=8, lr=3e-4, warmup_steps=500):
    model = model.to(DEVICE)

    dataset = MusicDataset(songs)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(epochs):
        total = 0

        for (x_tok, x_rel, x_sus, x_time), (y_pc, y_po, y_v, y_d, y_sus) in loader:
            x_tok = x_tok.to(DEVICE)
            x_rel = x_rel.to(DEVICE)
            x_sus = x_sus.to(DEVICE)
            x_time = x_time.to(DEVICE)
            y_pc, y_po, y_v, y_d, y_sus = (
                y_pc.to(DEVICE), y_po.to(DEVICE),
                y_v.to(DEVICE),  y_d.to(DEVICE), y_sus.to(DEVICE)
            )

            logits = model(x_tok, x_rel, x_sus, x_time, pc=y_pc, po=y_po, v=y_v, dt=y_d)
            loss   = loss_fn(logits, (y_pc, y_po, y_v, y_d, y_sus))

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            total += loss.item()

        print(f"tr1 Epoch {epoch+1} | loss {total:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt"))

    return model


def loadModel():
    model_path = os.path.join(MODEL_DIR, f"pretrained{MODEL_NUM}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained TransformerT1 model found at {model_path}")

    model = MusicTransformerT1()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def fineTune(model, song, seq_len=64, epochs=2, batch_size=4, lr=1e-5):

    model = model.to(DEVICE)
    model.train()

    # dataset built from ONLY this song (transposition augmentation still applies)
    dataset = MusicDataset([song], seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # smaller, constant LR than pretraining — no warmup/cosine schedule needed
    # for a short fine-tune run
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0

        for (x_tok, x_rel, x_sus, x_time), (y_pc, y_po, y_v, y_d, y_sus) in loader:
            x_tok = x_tok.to(DEVICE)
            x_rel = x_rel.to(DEVICE)
            x_sus = x_sus.to(DEVICE)
            x_time = x_time.to(DEVICE)
            y_pc, y_po, y_v, y_d, y_sus = (
                y_pc.to(DEVICE), y_po.to(DEVICE),
                y_v.to(DEVICE), y_d.to(DEVICE), y_sus.to(DEVICE)
            )

            logits = model(x_tok, x_rel, x_sus, x_time, pc=y_pc, po=y_po, v=y_v, dt=y_d)
            loss = loss_fn(logits, (y_pc, y_po, y_v, y_d, y_sus))

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            total += loss.item()

        print(f"[fine-tune] epoch {epoch+1} | loss {total:.4f}")

    return model


@torch.no_grad()
def compose(model, seedSong, targetSeconds=TARGET_SECONDS, temperature=1.0, top_p=0.9):
    model.eval()

    seedSong = seedSong[:SEED_NOTES]

    tokens  = torch.tensor(
        [encode_token(n[0], n[1], n[2]) for n in seedSong],
        dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    pitches = torch.tensor(
        [n[0] for n in seedSong],
        dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    sustains = torch.tensor(
        [n[3] for n in seedSong],
        dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    # elapsed time (dt bins converted to approx seconds) since the start of the OUTPUT
    # piece, not the original song's own timeline - keeps a single self-consistent 0..1
    # scale across the seed and everything generated after it
    seconds_per_bin = DT_MAX_SECONDS / MAX_TIME
    elapsed = 0.0
    seed_elapsed = [0.0]
    for n in seedSong[1:]:
        elapsed += n[2] * seconds_per_bin
        seed_elapsed.append(elapsed)

    times = torch.tensor(
        [min(1.0, t / targetSeconds) for t in seed_elapsed],
        dtype=torch.float32, device=DEVICE
    ).unsqueeze(0).unsqueeze(-1)

    def sample(logits):
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)  # [1, vocab]
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # zero out tokens whose cumulative mass exceeds top_p
        sorted_probs[cumsum - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        chosen = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, chosen).item()

    while elapsed < targetSeconds and tokens.size(1) < MAX_SEQ_LEN:

        rel = torch.zeros_like(pitches)
        rel[:, 1:] = pitches[:, 1:] - pitches[:, :-1]
        rel = torch.clamp(rel + 64, 0, 127)

        h = model.encode(tokens, rel, sustains, times)
        h_last = h[:, -1:, :]  # [1, 1, d_model] — only the next note matters

        pc_logits = model.out_pc(h_last)
        pc = sample(pc_logits)                                              # 0-11
        pc_t = torch.tensor([[pc]], dtype=torch.long, device=DEVICE)
        h_po = torch.cat([h_last, model.pc_cond(pc_t)], dim=-1)

        po_logits = model.out_po(h_po)
        po = sample(po_logits)                                              # 0-10
        po_t = torch.tensor([[po]], dtype=torch.long, device=DEVICE)
        h_v = torch.cat([h_po, model.po_cond(po_t)], dim=-1)

        v_logits = model.out_v(h_v)
        vel = sample(v_logits)                                              # 0-8
        v_t = torch.tensor([[vel]], dtype=torch.long, device=DEVICE)
        h_d = torch.cat([h_v, model.v_cond(v_t)], dim=-1)

        d_logits = model.out_d(h_d)
        dt = sample(d_logits)                                               # 0 to DT_VOCAB-1
        d_t = torch.tensor([[dt]], dtype=torch.long, device=DEVICE)
        h_sus = torch.cat([h_d, model.d_cond(d_t)], dim=-1)

        sus_logits = model.out_sus(h_sus)
        sus = sample(sus_logits)                                            # 0-1

        pitch = min(pc + po * 12, 127)

        elapsed += dt * seconds_per_bin
        time_value = min(1.0, elapsed / targetSeconds)

        next_tok   = torch.tensor([[encode_token(pitch, vel, dt)]], dtype=torch.long, device=DEVICE)
        next_pitch = torch.tensor([[pitch]], dtype=torch.long, device=DEVICE)
        next_sus   = torch.tensor([[sus]], dtype=torch.long, device=DEVICE)
        next_time  = torch.tensor([[[time_value]]], dtype=torch.float32, device=DEVICE)

        tokens   = torch.cat([tokens,   next_tok],   dim=1)
        pitches  = torch.cat([pitches,  next_pitch], dim=1)
        sustains = torch.cat([sustains, next_sus],   dim=1)
        times    = torch.cat([times,    next_time],  dim=1)

    decoded = [decode_token(t) for t in tokens.squeeze(0).tolist()]
    sus_list = sustains.squeeze(0).tolist()
    return [(p, v, d, s) for (p, v, d), s in zip(decoded, sus_list)]


def trainModel(songs):
    model = MusicTransformerT1()
    train(model, songs, epochs=5)

    return model


# def composeMusic(seedSong):
#     parser = Parser.MidiParser(MAX_VELOCITY, MAX_TIME, MAX_DURATION)
#
#     model = loadModel()
#     model = fineTune(model, seedSong, epochs=2, lr=1e-5)
#
#     generated = compose(model, seedSong)
#     # generatedNotes = parser.convertedNotes(generated)
#     # midi_tester.testMidi(generatedNotes, "midiTransformerT1.mid")