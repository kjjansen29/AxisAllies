import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLAYERS = 9

PHASES = [
    "tech",                           # 0
    "tech_chart_selection",           # 1
    "unit_repair",                    # 2
    "repair_quantity",                # 3
    "purchase_type",                  # 4
    "purchase_quantity",              # 5
    "kamikaze_quantity",              # 6
    "kamikaze_location",              # 7
    "kamikaze_target",                # 8
    "scramble_source",                # 9
    "scramble_unit_dest",             # 10
    "scramble_quantity",              # 11
    "bombard_source",                 # 12
    "bombard_dest",                   # 13
    "bombard_quantity",               # 14
    "declare_war",                    # 15
    "combat_move_source",             # 16
    "load_units_combat",              # 17
    "combat_move_unit_dest",          # 18
    "combat_move_quantity",           # 19
    "strategic_bombing_source",       # 20
    "strategic_bombing_decision",     # 21
    "strategic_bombing_quantity",     # 22
    "submarine_action_source",        # 23
    "submarine_action_type",          # 24
    "submarine_action_quantity",      # 25
    "casualty_type",                  # 26
    "casualty_quantity",              # 27
    "continue_or_retreat",            # 28
    "noncombat_source",               # 29
    "load_units_noncombat",           # 30
    "noncombat_unit_dest",            # 31
    "noncombat_quantity",             # 32
    "placement_territory",            # 33
    "placement_quantity",             # 34
    "sbr_interceptor_commitment",     # 35
    "sbr_interceptor_quantity",       # 36
    "sbr_air_battle_casualty_type",   # 37
    "sbr_air_battle_casualty_quantity", # 38
    "sbr_escort_commitment",          # 39
    "sbr_escort_quantity",            # 40
    "placement_carrier",              # 41
    "air_unit_land_on_carrier",            # 42
    "air_unit_land_on_carrier_dest",       # 43
    "air_unit_land_on_carrier_plane_qty",  # 44
    "air_unit_land_on_carrier_count",      # 45
]

# 16 consolidated policy heads.
# Head6581: slot_id * 329 + territory; slot_id 0-19; Pass=6580
POLICY_HEADS = [
    ("head2",    2),
    ("head3",    3),
    ("head7",    7),
    ("head10",   10),
    ("head13",   13),
    ("head14",   14),
    ("head20",   20),
    ("head49",   49),
    ("head128",  128),
    ("head202",  202),
    ("head330",  330),
    ("head6581", 6581),
    ("head988",  988),
]

HEAD_NAMES = [h[0] for h in POLICY_HEADS]
HEAD_SIZES = {h[0]: h[1] for h in POLICY_HEADS}

# Must match GetPolicyHeadForPhase in AIInferenceSpec.h
PHASE_TO_HEAD = {
    "tech":                             "head7",
    "tech_chart_selection":             "head2",
    "unit_repair":                      "head988",
    "repair_quantity":                  "head20",
    "purchase_type":                    "head20",
    "purchase_quantity":                "head20",
    "kamikaze_quantity":                "head7",
    "kamikaze_location":                "head128",
    "kamikaze_target":                  "head20",
    "scramble_source":                  "head6581",
    "scramble_unit_dest":               "head128",
    "scramble_quantity":                "head3",
    "bombard_source":                   "head6581",
    "bombard_dest":                     "head202",
    "bombard_quantity":                 "head20",
    "declare_war":                      "head10",
    "combat_move_source":               "head6581",
    "load_units_combat":                "head49",
    "combat_move_unit_dest":            "head330",
    "combat_move_quantity":             "head20",
    "strategic_bombing_source":         "head6581",
    "strategic_bombing_decision":       "head3",
    "strategic_bombing_quantity":       "head20",
    "submarine_action_source":          "head6581",
    "submarine_action_type":            "head2",
    "submarine_action_quantity":        "head20",
    "casualty_type":                    "head20",
    "casualty_quantity":                "head20",
    "continue_or_retreat":              "head13",
    "noncombat_source":                 "head6581",
    "load_units_noncombat":             "head49",
    "noncombat_unit_dest":              "head330",
    "noncombat_quantity":               "head20",
    "placement_territory":              "head330",
    "placement_quantity":               "head20",
    "placement_carrier":                "head6581",
    "air_unit_land_on_carrier":              "head6581",
    "air_unit_land_on_carrier_dest":         "head20",
    "air_unit_land_on_carrier_plane_qty":    "head20",
    "air_unit_land_on_carrier_count":        "head20",
    "sbr_interceptor_commitment":       "head6581",
    "sbr_interceptor_quantity":         "head20",
    "sbr_air_battle_casualty_type":     "head20",
    "sbr_air_battle_casualty_quantity": "head20",
    "sbr_escort_commitment":            "head6581",
    "sbr_escort_quantity":              "head20",
}

PHASE_ACTION_SIZES = {p: HEAD_SIZES[PHASE_TO_HEAD[p]] for p in PHASES}

# All global features (0-592) are read as board context.
GLOBAL_FEATURE_COUNT = 593

# The ONNX model is exported with this fixed batch size so the GPU runtime
# can fully optimize it. Must match INFERENCE_BATCH_SIZE in AIInferenceSpec.h.
EXPORT_BATCH_SIZE = 32


class RDGLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias   = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.weight * x_norm + self.bias


class EntitySelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q     = nn.Linear(hidden_dim, hidden_dim)
        self.k     = nn.Linear(hidden_dim, hidden_dim)
        self.v     = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.norm1 = RDGLayerNorm(hidden_dim)
        self.norm2 = RDGLayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, pad_mask):
        q    = self.q(x)
        k    = self.k(x)
        v    = self.v(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale + pad_mask
        attn = F.softmax(attn, dim=-1)
        x    = self.norm1(x + torch.matmul(attn, v))
        x    = self.norm2(x + self.ff(x))
        return x


class UnitEntityEncoder(nn.Module):
    def __init__(self, entity_feature_dim=25, entity_hidden_dim=64, num_layers=2):
        super().__init__()
        self.entity_hidden_dim = entity_hidden_dim
        self.input_proj        = nn.Linear(entity_feature_dim, entity_hidden_dim)
        self.layers            = nn.ModuleList([
            EntitySelfAttention(entity_hidden_dim) for _ in range(num_layers)
        ])
        self.empty_token = nn.Parameter(torch.randn(1, 1, entity_hidden_dim))
        self.register_buffer(
            "entity_idx",
            torch.arange(20, dtype=torch.float32).reshape(1, 20),
            persistent=False
        )

    def forward(self, entities, entity_counts):
        N = 329
        E = 20
        D = self.entity_hidden_dim
        entity_counts = entity_counts.reshape(-1, N)
        x             = self.input_proj(entities.reshape(-1, E, 25))
        counts_flat   = entity_counts.reshape(-1)
        valid         = (counts_flat.reshape(-1, 1) - self.entity_idx).clamp(0.0, 1.0)
        pad_mask      = (1.0 - valid.reshape(-1, 1, E)) * -1e9
        for layer in self.layers:
            x = layer(x, pad_mask)
        valid_3d  = valid.reshape(-1, E, 1)
        pooled    = (x * valid_3d).sum(dim=1)
        denom     = counts_flat.clamp(min=1.0).reshape(-1, 1)
        pooled    = pooled / denom
        pooled    = pooled.reshape(-1, N, D)
        has_units = entity_counts.clamp(0.0, 1.0).reshape(-1, N, 1)
        pooled    = pooled * has_units + self.empty_token * (1.0 - has_units)
        return pooled


class EdgeAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q     = nn.Linear(hidden_dim, hidden_dim)
        self.k     = nn.Linear(hidden_dim, hidden_dim)
        self.v     = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, nodes, adjacency_matrix):
        q    = self.q(nodes)
        k    = self.k(nodes)
        v    = self.v(nodes)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = (1.0 - adjacency_matrix) * -1e9
        attn = F.softmax(attn + mask, dim=-1)
        return torch.matmul(attn, v)


class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn  = EdgeAttentionLayer(hidden_dim)
        self.norm1 = RDGLayerNorm(hidden_dim)
        self.norm2 = RDGLayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, adjacency_matrix):
        x = self.norm1(x + self.attn(x, adjacency_matrix))
        x = self.norm2(x + self.ff(x))
        return x


class AxisAlliesGraphNet(nn.Module):
    def __init__(self, adjacency_matrix: torch.Tensor, hidden_dim=192, num_layers=6):
        super().__init__()
        self.num_players      = NUM_PLAYERS
        self.num_phases       = len(PHASES)
        self.model_node_count = 329
        self.total_node_count = self.model_node_count + 1

        if adjacency_matrix is None:
            raise RuntimeError("Adjacency matrix must be provided.")
        if adjacency_matrix.shape != (self.model_node_count, self.model_node_count):
            raise RuntimeError(
                f"Invalid adjacency shape: expected "
                f"{(self.model_node_count, self.model_node_count)}, "
                f"got {tuple(adjacency_matrix.shape)}"
            )

        adj_with_global = torch.ones(
            (self.total_node_count, self.total_node_count), dtype=torch.float32)
        adj_with_global[1:, 1:] = adjacency_matrix.float()
        adj_with_global[0, 0]   = 1.0
        self.register_buffer(
            "adjacency_matrix",
            adj_with_global.reshape(1, self.total_node_count, self.total_node_count),
            persistent=True
        )

        self.node_feature_dim   = 20
        self.entity_feature_dim = 25
        self.entity_hidden_dim  = 64
        self.max_entities       = 20
        self.global_feature_dim = GLOBAL_FEATURE_COUNT
        # Every global feature is board context
        self.board_feature_dim  = GLOBAL_FEATURE_COUNT
        self.hidden_dim         = hidden_dim

        self.entity_encoder = UnitEntityEncoder(
            entity_feature_dim=self.entity_feature_dim,
            entity_hidden_dim=self.entity_hidden_dim,
            num_layers=2
        )

        # Territory projection: [20 + 64] = 84 → hidden_dim
        self.territory_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.entity_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            GraphTransformerBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Global board context -> one hidden vector, added to the global
        # token and to every territory node. (Previously a single Linear
        # to 330 * hidden_dim outputs, ~37.5M weights; this is ~150K.)
        self.global_proj = nn.Sequential(
            nn.Linear(self.board_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 16 consolidated policy heads
        self.policy_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, size)
            for name, size in POLICY_HEADS
        })

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_players),
            nn.Tanh()
        )

    def forward(self, node_features, global_features, unit_entities, entity_counts):
        entity_repr     = self.entity_encoder(unit_entities, entity_counts)
        territory_input = torch.cat([node_features, entity_repr], dim=-1)
        x               = self.territory_proj(territory_input)

        # Board context = every global feature
        gf = self.global_proj(global_features).reshape(-1, 1, self.hidden_dim)
        g  = self.global_token + gf
        x  = x + gf
        x  = torch.cat([g, x], dim=1)

        for layer in self.layers:
            x = layer(x, self.adjacency_matrix)

        global_repr = x[:, 0:1, :].reshape(-1, self.hidden_dim)
        node_repr   = x[:, 1:, :].sum(dim=1) * (1.0 / 329.0)
        h           = global_repr + node_repr

        outputs = {
            name: self.policy_heads[name](h)
            for name in HEAD_NAMES
        }
        outputs["value"] = self.value_head(h)
        return outputs

    def export_onnx(self, path, device):
        self.eval()
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        cpu_model = self.cpu()
        cpu_model.eval()

        B = EXPORT_BATCH_SIZE
        N = self.model_node_count
        E = self.max_entities

        dummy_nodes    = torch.zeros((B, N, self.node_feature_dim),     dtype=torch.float32)
        dummy_global   = torch.zeros((B, self.global_feature_dim),       dtype=torch.float32)
        dummy_entities = torch.zeros((B, N, E, self.entity_feature_dim), dtype=torch.float32)
        dummy_counts   = torch.zeros((B, N, 1),                          dtype=torch.float32)

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        output_names = HEAD_NAMES + ["value"]
        # No dynamic axes: every input and output has the fixed batch size.

        with torch.no_grad():
            torch.onnx.export(
                cpu_model,
                (dummy_nodes, dummy_global, dummy_entities, dummy_counts),
                abs_path,
                opset_version=16,
                export_params=True,
                do_constant_folding=True,
                input_names=["node_features", "global_features",
                             "unit_entities", "entity_counts"],
                output_names=output_names,
            )

        self.to(device)
        if not os.path.exists(abs_path):
            raise RuntimeError(f"ONNX export failed: {abs_path}")
        if os.path.getsize(abs_path) < 2048:
            raise RuntimeError(f"ONNX export too small: {abs_path}")
        return abs_path

    def save(self, path):
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        torch.save(self.state_dict(), abs_path)
        return abs_path


def load_compatible_weights(model, pt_path, device):
    """Load saved weights, skipping any layer whose shape has changed
    (e.g. global_proj after the global-feature input change)."""
    saved   = torch.load(pt_path, map_location=device, weights_only=True)
    current = model.state_dict()
    kept, skipped = {}, []
    for k, v in saved.items():
        if k in current and current[k].shape == v.shape:
            kept[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(kept, strict=False)
    if skipped:
        print(f"[train] Skipped {len(skipped)} tensors with changed shape: {skipped}")


# ================================================================
# REPLAY DATA (columnar, NumPy)
#
# All samples are held as a set of NumPy arrays ("columns"). Sparse
# data (policy, legal actions, entities) is stored as flat arrays plus
# per-sample offsets: sample i owns entries [off[i], off[i + 1]).
#
# Readers:
#   .bin    binary shard written by UAI_ReplayBufferManager (Stage B)
#   .jsonl  one JSON sample per line (Stage A shards)
#   .json   legacy TotalReplayBuffer.json
# ================================================================
import zlib
import numpy as np

NODE_SIZE        = 329 * 20
ENTITY_FEATURES  = 25
MAX_ENTITIES     = 20
PHASE_HEAD_INDEX = np.array(
    [HEAD_NAMES.index(PHASE_TO_HEAD[p]) for p in PHASES], dtype=np.int64)

_COLUMNS = ["phase", "player", "policy_size", "visit", "value", "node", "glob",
            "pol_off", "pol_idx", "pol_val", "leg_off", "leg_idx",
            "ent_off", "ent_t", "ent_f"]


def _empty_chunk():
    return {
        "phase":       np.zeros(0, np.int32),
        "player":      np.zeros(0, np.int32),
        "policy_size": np.zeros(0, np.int32),
        "visit":       np.zeros(0, np.float32),
        "value":       np.zeros((0, NUM_PLAYERS), np.float32),
        "node":        np.zeros((0, NODE_SIZE), np.float32),
        "glob":        np.zeros((0, GLOBAL_FEATURE_COUNT), np.float32),
        "pol_off":     np.zeros(1, np.int64),
        "pol_idx":     np.zeros(0, np.int32),
        "pol_val":     np.zeros(0, np.float32),
        "leg_off":     np.zeros(1, np.int64),
        "leg_idx":     np.zeros(0, np.int32),
        "ent_off":     np.zeros(1, np.int64),
        "ent_t":       np.zeros(0, np.int32),
        "ent_f":       np.zeros((0, ENTITY_FEATURES), np.float32),
    }


def _concat_chunks(chunks):
    chunks = [c for c in chunks if len(c["phase"]) > 0]
    if not chunks:
        return _empty_chunk()
    out = {}
    for key in ["phase", "player", "policy_size", "visit", "value", "node", "glob",
                "pol_idx", "pol_val", "leg_idx", "ent_t", "ent_f"]:
        out[key] = np.concatenate([c[key] for c in chunks])
    for off, data in [("pol_off", "pol_idx"), ("leg_off", "leg_idx"), ("ent_off", "ent_t")]:
        parts, base = [np.zeros(1, np.int64)], 0
        for c in chunks:
            parts.append(c[off][1:].astype(np.int64) + base)
            base += len(c[data])
        out[off] = np.concatenate(parts)
    return out


def _decompress(blob):
    for wbits in (15, -15, 31):   # zlib, raw deflate, gzip
        try:
            return zlib.decompress(blob, wbits)
        except zlib.error:
            continue
    raise RuntimeError("Could not decompress binary shard")


def _read_bin_shard(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"AAR1":
        raise RuntimeError(f"Not a binary shard: {path}")
    version, raw_size, comp_size = np.frombuffer(data, dtype="<u4", count=3, offset=4)
    if version != 1:
        raise RuntimeError(f"Unsupported shard version {version}: {path}")
    raw = _decompress(data[16:16 + int(comp_size)])
    if len(raw) != int(raw_size):
        raise RuntimeError(f"Shard size mismatch: {path}")

    pos = 0
    def take(dtype, count):
        nonlocal pos
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
        pos += arr.nbytes
        return arr

    n = int(take("<i4", 1)[0])
    c = {}
    c["phase"]       = take("<i4", n).astype(np.int32)
    c["player"]      = take("<i4", n).astype(np.int32)
    c["policy_size"] = take("<i4", n).astype(np.int32)
    c["visit"]       = take("<f4", n).astype(np.float32)
    c["value"]       = take("<f4", n * NUM_PLAYERS).reshape(n, NUM_PLAYERS).astype(np.float32)
    c["node"]        = take("<f4", n * NODE_SIZE).reshape(n, NODE_SIZE).astype(np.float32)
    c["glob"]        = take("<f4", n * GLOBAL_FEATURE_COUNT).reshape(
        n, GLOBAL_FEATURE_COUNT).astype(np.float32)
    c["pol_off"] = take("<i4", n + 1).astype(np.int64)
    c["pol_idx"] = take("<i4", int(c["pol_off"][-1])).astype(np.int32)
    c["pol_val"] = take("<f4", int(c["pol_off"][-1])).astype(np.float32)
    c["leg_off"] = take("<i4", n + 1).astype(np.int64)
    c["leg_idx"] = take("<i4", int(c["leg_off"][-1])).astype(np.int32)
    c["ent_off"] = take("<i4", n + 1).astype(np.int64)
    m = int(c["ent_off"][-1])
    c["ent_t"] = take("<i4", m).astype(np.int32)
    c["ent_f"] = take("<f4", m * ENTITY_FEATURES).reshape(m, ENTITY_FEATURES).astype(np.float32)
    return c


def _dicts_to_chunk(samples):
    """Convert JSON samples (Stage A sparse or legacy dense) to a chunk."""
    n = len(samples)
    if n == 0:
        return _empty_chunk()
    c = _empty_chunk()
    c["phase"]  = np.array([int(s["phase_id"]) for s in samples], np.int32)
    c["player"] = np.array([int(s["player_id"]) for s in samples], np.int32)
    c["visit"]  = np.array([float(s.get("visit_count", 1.0)) for s in samples], np.float32)
    c["value"]  = np.array([s["value_target"] for s in samples], np.float32).reshape(n, NUM_PLAYERS)
    c["node"]   = np.array([s["node_features"] for s in samples], np.float32).reshape(n, NODE_SIZE)
    c["glob"]   = np.array([s["global_features"] for s in samples], np.float32).reshape(
        n, GLOBAL_FEATURE_COUNT)

    sizes, p_off, p_idx, p_val = [], [0], [], []
    l_off, l_idx = [0], []
    e_off, e_t, e_f = [0], [], []
    for s in samples:
        if "policy" in s:
            sizes.append(int(s["policy_size"]))
            for i, p in s["policy"]:
                if p > 0:
                    p_idx.append(int(i)); p_val.append(float(p))
        else:
            dense = s["policy_target"]
            sizes.append(len(dense))
            for i, p in enumerate(dense):
                if p > 0:
                    p_idx.append(i); p_val.append(float(p))
        p_off.append(len(p_idx))

        l_idx.extend(int(a) for a in s.get("legal_actions", []))
        l_off.append(len(l_idx))

        if "entities" in s:
            terr = [(int(t), ents) for t, ents in s["entities"]]
        else:
            terr = [(t, terr.get("entities", [])[:int(terr.get("count", 0))])
                    for t, terr in enumerate(s.get("entity_list", []))]
        for t, ents in sorted(terr, key=lambda x: x[0]):
            for feats in ents:
                e_t.append(t); e_f.append(feats[:ENTITY_FEATURES])
        e_off.append(len(e_t))

    c["policy_size"] = np.array(sizes, np.int32)
    c["pol_off"] = np.array(p_off, np.int64)
    c["pol_idx"] = np.array(p_idx, np.int32)
    c["pol_val"] = np.array(p_val, np.float32)
    c["leg_off"] = np.array(l_off, np.int64)
    c["leg_idx"] = np.array(l_idx, np.int32)
    c["ent_off"] = np.array(e_off, np.int64)
    c["ent_t"]   = np.array(e_t, np.int32)
    c["ent_f"]   = np.array(e_f, np.float32).reshape(-1, ENTITY_FEATURES)
    return c


def _read_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _read_legacy_total(path):
    """Legacy format: [{"PhaseId": P, "samples": [...]}, ...]"""
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    samples = []
    for b in root:
        for s in b["samples"]:
            s.setdefault("phase_id", int(b["PhaseId"]))
            samples.append(s)
    return samples


def load_training_dataset(dataset_path):
    """dataset_path is the replay index (Saved/AITraining/Replay/index.json).
    A legacy TotalReplayBuffer.json can also be passed directly.
    Returns one columnar chunk holding every sample, oldest first."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    if isinstance(root, list):
        return _dicts_to_chunk(_read_legacy_total(dataset_path))

    base_dir = os.path.dirname(os.path.abspath(dataset_path))
    chunks = []
    for shard in root.get("shards", []):
        path = os.path.normpath(os.path.join(base_dir, shard["file"]))
        if not os.path.exists(path):
            print(f"[train] Missing shard skipped: {path}")
            continue
        if path.endswith(".bin"):
            chunks.append(_read_bin_shard(path))
        elif path.endswith(".jsonl"):
            chunks.append(_dicts_to_chunk(_read_jsonl(path)))
        else:
            chunks.append(_dicts_to_chunk(_read_legacy_total(path)))
    return _concat_chunks(chunks)


def compute_sample_scores(data, temperature=1.0):
    n = len(data["phase"])
    if n == 0:
        return np.zeros(0)
    visit_score = np.minimum(data["visit"].astype(np.float64), 100.0) + 1.0

    # Policy entropy per sample (nonzero entries only)
    counts  = np.diff(data["pol_off"])
    seg     = np.repeat(np.arange(n), counts)
    vals    = data["pol_val"].astype(np.float64)
    sums    = np.bincount(seg, weights=vals, minlength=n)
    norm    = vals / np.maximum(sums[seg], 1e-12)
    entropy = np.bincount(seg, weights=-norm * np.log(norm + 1e-10), minlength=n)
    entropy[sums <= 1e-8] = 0.0
    entropy_score = np.maximum(entropy, 0.0) + 0.1

    recency_score = 0.5 + 0.5 * (np.arange(n) / max(n - 1, 1))
    scores = (visit_score ** 0.5) * (entropy_score ** 0.3) * (recency_score ** 0.2)
    if temperature != 1.0:
        scores = scores ** (1.0 / temperature)
    total = scores.sum()
    return scores / total if total > 1e-10 else np.full(n, 1.0 / n)


def select_training_batch(num_samples, batch_size, probs):
    n = min(batch_size, num_samples)
    return np.random.choice(num_samples, size=n, replace=True, p=probs)


def build_batch_arrays(data, idx):
    """Build one training batch as NumPy arrays (no torch needed)."""
    B = len(idx)
    out = {
        "node_features":   data["node"][idx].reshape(B, 329, 20),
        "global_features": data["glob"][idx],
        "values":          data["value"][idx],
        "phase_ids":       data["phase"][idx].astype(np.int64),
        "player_ids":      data["player"][idx].astype(np.int64),
        "policy_targets":  {},
        "policy_masks":    {},
    }

    # ---- Policy targets and legal masks, per head, in batch order ----
    heads = PHASE_HEAD_INDEX[out["phase_ids"]]
    for h, name in enumerate(HEAD_NAMES):
        size = HEAD_SIZES[name]
        rows = np.nonzero(heads == h)[0]
        targets = np.zeros((len(rows), size), np.float32)
        legal   = np.zeros((len(rows), size), np.bool_)
        for r, j in enumerate(rows):
            i = idx[j]
            if int(data["policy_size"][i]) != size:
                raise RuntimeError(
                    f"Phase {PHASES[data['phase'][i]]}: policy size "
                    f"{int(data['policy_size'][i])}, expected {size}")
            a, b = data["pol_off"][i], data["pol_off"][i + 1]
            pi = data["pol_idx"][a:b]
            ok = (pi >= 0) & (pi < size)
            targets[r, pi[ok]] = data["pol_val"][a:b][ok]
            a, b = data["leg_off"][i], data["leg_off"][i + 1]
            li = data["leg_idx"][a:b]
            legal[r, li[(li >= 0) & (li < size)]] = True
        legal |= targets > 0.0   # older samples without legal actions
        out["policy_targets"][name] = targets
        out["policy_masks"][name]   = legal

    # ---- Entities: ordered by territory, then slot ----
    unit   = np.zeros((B, 329, MAX_ENTITIES, ENTITY_FEATURES), np.float32)
    counts = np.zeros((B, 329), np.float32)
    for j, i in enumerate(idx):
        a, b = data["ent_off"][i], data["ent_off"][i + 1]
        if a == b:
            continue
        ts    = data["ent_t"][a:b]
        feats = data["ent_f"][a:b]
        _, first, cnt = np.unique(ts, return_index=True, return_counts=True)
        slots = np.arange(len(ts)) - np.repeat(first, cnt)
        keep  = (slots < MAX_ENTITIES) & (ts >= 0) & (ts < 329)
        unit[j, ts[keep], slots[keep]] = feats[keep]
        counts[j] = np.minimum(np.bincount(ts[keep], minlength=329)[:329], MAX_ENTITIES)
    out["unit_entities"] = unit
    out["entity_counts"] = counts
    return out


def build_batch(data, idx, device):
    arr = build_batch_arrays(data, idx)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return {
        "node_features":   t(arr["node_features"]),
        "global_features": t(arr["global_features"]),
        "unit_entities":   t(arr["unit_entities"]),
        "entity_counts":   t(arr["entity_counts"]),
        "values":          t(arr["values"]),
        "phase_ids":       t(arr["phase_ids"]),
        "player_ids":      t(arr["player_ids"]),
        "policy_targets":  {k: t(v) for k, v in arr["policy_targets"].items()},
        "policy_masks":    {k: t(v) for k, v in arr["policy_masks"].items()},
    }


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        batch["node_features"], batch["global_features"],
        batch["unit_entities"], batch["entity_counts"]
    )
    total_policy_loss    = 0.0
    per_head_policy_loss = {}

    # Rows of each head's target/mask tensors are in batch order among
    # samples of that head, so select the matching rows of the model output.
    for head_name in HEAD_NAMES:
        head_phases = [i for i, p in enumerate(PHASES) if PHASE_TO_HEAD[p] == head_name]
        mask = torch.zeros(
            batch["phase_ids"].shape[0], dtype=torch.bool,
            device=batch["phase_ids"].device)
        for pid in head_phases:
            mask |= (batch["phase_ids"] == pid)
        if mask.sum() == 0:
            continue
        preds      = outputs[head_name][mask]
        targets    = batch["policy_targets"][head_name]
        legal_mask = batch["policy_masks"][head_name]
        has_target = targets.sum(dim=1) > 0.0
        if has_target.sum() == 0:
            continue
        preds        = preds[has_target].masked_fill(~legal_mask[has_target], -1e9)
        targets      = targets[has_target]
        logp         = F.log_softmax(preds, dim=1)
        norm_targets = targets / targets.sum(dim=1, keepdim=True)
        head_loss    = -(norm_targets * logp).sum(dim=1).mean()
        total_policy_loss += head_loss
        per_head_policy_loss[head_name] = head_loss.item()
    total_value_loss = F.mse_loss(outputs["value"], batch["values"])
    loss = total_policy_loss + total_value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss":                 loss.item(),
        "policy_loss":          total_policy_loss.item()
                                if hasattr(total_policy_loss, 'item')
                                else float(total_policy_loss),
        "value_loss":           total_value_loss.item(),
        "per_head_policy_loss": per_head_policy_loss
    }


def load_adjacency_matrix_from_json(path: str, num_nodes: int = 329):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for a, b in data["edges"]:
        matrix[a][b] = 1.0
        matrix[b][a] = 1.0
    return matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(device):
    base_dir         = os.path.dirname(os.path.abspath(__file__))
    adj_path         = os.path.join(base_dir, "AITerritoryGraph.json")
    adjacency_matrix = load_adjacency_matrix_from_json(adj_path, 329)
    return AxisAlliesGraphNet(adjacency_matrix).to(device)


def ensure_models_exist():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    pt_path   = os.path.join(base_dir, "axis_allies.pt")
    onnx_path = os.path.join(base_dir, "axis_allies.onnx")
    os.makedirs(base_dir, exist_ok=True)
    model = create_model(device)
    if os.path.exists(pt_path):
        # Re-export the ONNX from the existing weights (e.g. after deleting
        # only the .onnx to change the export settings).
        load_compatible_weights(model, pt_path, device)
    model.eval()
    if not os.path.exists(pt_path):
        model.save(pt_path)
    if not os.path.exists(onnx_path):
        model.export_onnx(onnx_path, device)
    return True


if __name__ == "__main__":
    import argparse, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default=None)
    parser.add_argument("--out_dir",    type=str,   default=None)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--steps",      type=int,   default=100)
    args = parser.parse_args()

    if args.dataset is None:
        ensure_models_exist()
    else:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        out_dir   = args.out_dir if args.out_dir else \
            os.path.dirname(os.path.abspath(__file__))
        os.makedirs(out_dir, exist_ok=True)
        pt_path   = os.path.join(out_dir, "axis_allies.pt")
        onnx_path = os.path.join(out_dir, "axis_allies.onnx")

        print(f"[train] Loading dataset: {args.dataset}")
        data          = load_training_dataset(args.dataset)
        total_samples = len(data["phase"])
        print(f"[train] Total samples: {total_samples}")
        if total_samples == 0:
            raise RuntimeError("Dataset contains no samples")

        probs = compute_sample_scores(data, temperature=1.0)

        model = create_model(device)
        if os.path.exists(pt_path):
            print(f"[train] Loading existing weights: {pt_path}")
            load_compatible_weights(model, pt_path, device)

        model.train()
        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
        batch_size = min(args.batch_size, total_samples)
        steps      = min(
            max(args.steps, (total_samples * 4) // max(batch_size, 1)), 2000)
        print(f"[train] batch_size={batch_size} steps={steps} device={device}")

        for step in range(steps):
            idx      = select_training_batch(total_samples, batch_size, probs)
            batch    = build_batch(data, idx, device)
            result   = train_step(model, batch, optimizer)
            if (step + 1) % max(steps // 10, 1) == 0:
                print(
                    f"[train] step={step+1}/{steps} "
                    f"loss={result['loss']:.6f} "
                    f"policy={result['policy_loss']:.6f} "
                    f"value={result['value_loss']:.6f}"
                )
                per_head = result["per_head_policy_loss"]
                head_str = " ".join(
                    f"{h}={per_head[h]:.4f}" for h in HEAD_NAMES if h in per_head)
                print(f"[train]   per-head: {head_str}")

        model.eval()
        print(f"[train] Saved .pt:   {model.save(pt_path)}")
        print(f"[train] Saved .onnx: {model.export_onnx(onnx_path, device)}")