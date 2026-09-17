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
    "scramble_unit_dest":               "head20",
    "scramble_quantity":                "head3",
    "bombard_source":                   "head128",
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
        self.global_feature_dim = 593
        self.board_feature_dim  = 551
        self.hidden_dim         = hidden_dim

        self.entity_encoder = UnitEntityEncoder(
            entity_feature_dim=self.entity_feature_dim,
            entity_hidden_dim=self.entity_hidden_dim,
            num_layers=2
        )

        # Territory projection: [19 + 64] = 83 → hidden_dim
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

        self.global_proj = nn.Linear(
            self.board_feature_dim,
            self.total_node_count * hidden_dim
        )

        self.pending_action_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # 16 consolidated policy heads
        self.policy_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim + 32, size)
            for name, size in POLICY_HEADS
        })

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_players),
            nn.Tanh()
        )

    def forward(self, node_features, global_features, unit_entities, entity_counts):
        entity_repr     = self.entity_encoder(unit_entities, entity_counts)
        territory_input = torch.cat([node_features, entity_repr], dim=-1)
        x               = self.territory_proj(territory_input)

        board_features       = global_features[:, 0:551]
        pending_action_value = global_features[:, 551:552]

        gf = self.global_proj(board_features)
        gf = gf.reshape(-1, self.total_node_count, self.hidden_dim)
        g  = self.global_token + gf[:, 0:1, :]
        x  = x + gf[:, 1:, :]
        x  = torch.cat([g, x], dim=1)

        for layer in self.layers:
            x = layer(x, self.adjacency_matrix)

        global_repr = x[:, 0:1, :].reshape(-1, self.hidden_dim)
        node_repr   = x[:, 1:, :].sum(dim=1) * (1.0 / 329.0)
        h           = global_repr + node_repr

        pending_embed  = self.pending_action_embed(pending_action_value)
        h_with_pending = torch.cat([h, pending_embed], dim=1)

        outputs = {
            name: self.policy_heads[name](h_with_pending)
            for name in HEAD_NAMES
        }
        outputs["value"] = self.value_head(h_with_pending)
        return outputs

    def export_onnx(self, path, device):
        self.eval()
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        cpu_model = self.cpu()
        cpu_model.eval()

        B = 1
        N = self.model_node_count
        E = self.max_entities

        dummy_nodes    = torch.zeros((B, N, self.node_feature_dim),     dtype=torch.float32)
        dummy_global   = torch.zeros((B, self.global_feature_dim),       dtype=torch.float32)
        dummy_entities = torch.zeros((B, N, E, self.entity_feature_dim), dtype=torch.float32)
        dummy_counts   = torch.zeros((B, N, 1),                          dtype=torch.float32)

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        output_names = HEAD_NAMES + ["value"]
        dynamic_axes = {
            "node_features":   {0: "batch"},
            "global_features": {0: "batch"},
            "unit_entities":   {0: "batch"},
            "entity_counts":   {0: "batch"},
            "value":           {0: "batch"},
        }
        for name in HEAD_NAMES:
            dynamic_axes[name] = {0: "batch"}

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
                dynamic_axes=dynamic_axes,
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


def load_training_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    return [{"phase_id": int(b["PhaseId"]), "samples": b["samples"]} for b in root]


def _build_entity_tensors(samples_list, device, max_entities=20, entity_features=25):
    B = len(samples_list)
    N = 329
    unit_entities = torch.zeros(
        (B, N, max_entities, entity_features), dtype=torch.float32, device=device)
    entity_counts = torch.zeros((B, N), dtype=torch.float32, device=device)
    for i, s in enumerate(samples_list):
        for t, terr in enumerate(s.get("entity_list", [])):
            count = min(int(terr.get("count", 0)), max_entities)
            entity_counts[i, t] = float(count)
            for e_idx, feats in enumerate(terr.get("entities", [])[:count]):
                unit_entities[i, t, e_idx] = torch.tensor(
                    feats, dtype=torch.float32, device=device)
    return unit_entities, entity_counts


def flatten_dataset(dataset):
    flat = []
    for batch in dataset:
        phase_id   = int(batch["phase_id"])
        phase_name = PHASES[phase_id]
        head_name  = PHASE_TO_HEAD[phase_name]
        for s in batch["samples"]:
            flat.append((phase_name, head_name, s))
    return flat


def compute_sample_scores(dataset, temperature=1.0):
    import math
    scores        = []
    total_samples = sum(len(b["samples"]) for b in dataset)
    sample_idx    = 0
    for batch in dataset:
        for s in batch["samples"]:
            visit_score   = min(float(s.get("visit_count", 1.0)), 100.0) + 1.0
            policy        = s["policy_target"]
            policy_sum    = sum(policy)
            if policy_sum > 1e-8:
                norm    = [p / policy_sum for p in policy]
                entropy = -sum(p * math.log(p + 1e-10) for p in norm if p > 0)
            else:
                entropy = 0.0
            entropy_score = max(0.0, entropy) + 0.1
            recency_score = 0.5 + 0.5 * (sample_idx / max(total_samples - 1, 1))
            scores.append(
                (visit_score ** 0.5) * (entropy_score ** 0.3) * (recency_score ** 0.2))
            sample_idx += 1
    if temperature != 1.0:
        scores = [s ** (1.0 / temperature) for s in scores]
    total = sum(scores)
    return [s / total for s in scores] if total > 1e-10 \
        else [1.0 / len(scores)] * len(scores)


def select_training_batch(flat_samples, batch_size, probs):
    import random
    n       = min(batch_size, len(flat_samples))
    indices = random.choices(range(len(flat_samples)), weights=probs, k=n)
    return [flat_samples[i] for i in indices]


def build_batch_from_selected(selected_samples, device):
    node_features   = []
    global_features = []
    entity_samples  = []
    policy_targets  = {name: [] for name in HEAD_NAMES}
    values          = []
    phase_ids       = []
    player_ids      = []

    for phase_name, head_name, s in selected_samples:
        phase_id = PHASES.index(phase_name)
        node_features.append(
            torch.tensor(s["node_features"], dtype=torch.float32, device=device))
        global_features.append(
            torch.tensor(s["global_features"], dtype=torch.float32, device=device))
        values.append(
            torch.tensor(s["value_target"], dtype=torch.float32, device=device))
        entity_samples.append(s)
        pt = torch.tensor(s["policy_target"], dtype=torch.float32, device=device)
        if pt.shape[0] != HEAD_SIZES[head_name]:
            raise RuntimeError(
                f"Policy target size mismatch for phase {phase_name}: "
                f"expected {HEAD_SIZES[head_name]}, got {pt.shape[0]}")
        policy_targets[head_name].append(pt)
        phase_ids.append(phase_id)
        player_ids.append(int(s["player_id"]))

    stacked_nodes = torch.stack(node_features).reshape(-1, 329, 20)
    unit_entities, entity_counts = _build_entity_tensors(entity_samples, device)

    return {
        "node_features":   stacked_nodes,
        "global_features": torch.stack(global_features),
        "unit_entities":   unit_entities,
        "entity_counts":   entity_counts,
        "values":          torch.stack(values),
        "phase_ids":       torch.tensor(phase_ids, device=device),
        "player_ids":      torch.tensor(player_ids, device=device),
        "policy_targets":  {
            name: torch.stack(v) if len(v) else torch.empty(
                (0, HEAD_SIZES[name]), device=device)
            for name, v in policy_targets.items()
        }
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
        targets    = batch["policy_targets"][head_name][mask]
        legal_mask = targets > 0.0
        if legal_mask.sum() == 0:
            continue
        preds        = preds.masked_fill(~legal_mask, -1e9)
        logp         = F.log_softmax(preds, dim=1)
        target_sum   = targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        norm_targets = targets / target_sum
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
        dataset       = load_training_dataset(args.dataset)
        total_samples = sum(len(b["samples"]) for b in dataset)
        print(f"[train] Total samples: {total_samples}")

        probs        = compute_sample_scores(dataset, temperature=1.0)
        flat_samples = flatten_dataset(dataset)

        model = create_model(device)
        if os.path.exists(pt_path):
            print(f"[train] Loading existing weights: {pt_path}")
            state = torch.load(pt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)

        model.train()
        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
        batch_size = min(args.batch_size, total_samples)
        steps      = min(
            max(args.steps, (total_samples * 4) // max(batch_size, 1)), 2000)
        print(f"[train] batch_size={batch_size} steps={steps} device={device}")

        for step in range(steps):
            selected = select_training_batch(flat_samples, batch_size, probs)
            batch    = build_batch_from_selected(selected, device)
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