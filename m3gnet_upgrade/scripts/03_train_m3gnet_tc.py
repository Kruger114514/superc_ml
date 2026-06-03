"""
03_train_m3gnet_tc.py — 训练 M3GNet 预测 Tc
===========================================

用 matgl + PyTorch Lightning 训练 M3GNet 在 3DSC 上回归 Tc。

关键决策:
1. 从预训练的 formation energy 模型 fine-tune (而不是从头训)
2. 训练 log(Tc) 而不是 Tc 直接, 因为 Tc 跨多个量级 (0.001 K - 200 K)
3. 损失函数 MSLE
4. 按家族分层验证

需要 GPU. 配置见 configs/train_config.yaml.

运行: python scripts/03_train_m3gnet_tc.py
     或自定义 config: python scripts/03_train_m3gnet_tc.py --config xxx.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pymatgen.core import Structure
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
from matgl.models import M3GNet

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. 配置加载
# =============================================================================

def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# 2. 数据加载 (复用 02_prepare_dataset.py 的 pkl)
# =============================================================================

def load_pkl(path: Path) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def records_to_dataset(
    records: List[Dict],
    converter: Structure2Graph,
    log_transform: bool,
):
    """把 records (dict list) 转成 MGLDataset"""
    structures = [r["structure"] for r in records]
    tc_values = [r["tc"] for r in records]
    if log_transform:
        # log(Tc + 1) 避免 log(0)
        labels = {"tc": torch.tensor([np.log(t + 1) for t in tc_values], dtype=torch.float32)}
    else:
        labels = {"tc": torch.tensor(tc_values, dtype=torch.float32)}

    return MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels=labels,
        include_line_graph=True,
    )


# =============================================================================
# 3. Lightning Module (自定义训练逻辑)
# =============================================================================

class TcRegressor(L.LightningModule):
    """
    用 M3GNet 作为 backbone, 回归 Tc.
    """

    def __init__(
        self,
        model: M3GNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        log_transform: bool = True,
        loss_type: str = "msle",
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.log_transform = log_transform
        self.loss_type = loss_type
        self.save_hyperparameters(ignore=["model"])

    def forward(self, g, lg=None, state_attr=None):
        # M3GNet 输入: g (graph), lg (line graph), state_attr
        return self.model(g=g, l_g=lg, state_attr=state_attr)

    def _compute_loss(self, pred, target):
        if self.loss_type == "msle":
            # pred 和 target 都已经是 log 空间, 所以 MSLE = MSE on log
            return F.mse_loss(pred, target)
        elif self.loss_type == "mae":
            return F.l1_loss(pred, target)
        else:
            return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        g, lg, state_attr, labels = batch
        pred = self(g, lg, state_attr).squeeze(-1)
        target = labels["tc"]
        loss = self._compute_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True, batch_size=g.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        g, lg, state_attr, labels = batch
        pred = self(g, lg, state_attr).squeeze(-1)
        target = labels["tc"]
        loss = self._compute_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True, batch_size=g.batch_size)

        # 计算实际 Tc 空间的 MAE (用于人类可读的指标)
        if self.log_transform:
            pred_tc = torch.exp(pred) - 1
            true_tc = torch.exp(target) - 1
        else:
            pred_tc, true_tc = pred, target
        tc_mae = F.l1_loss(pred_tc, true_tc)
        self.log("val_tc_mae_K", tc_mae, prog_bar=True, batch_size=g.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        g, lg, state_attr, labels = batch
        pred = self(g, lg, state_attr).squeeze(-1)
        target = labels["tc"]
        if self.log_transform:
            pred_tc = torch.exp(pred) - 1
            true_tc = torch.exp(target) - 1
        else:
            pred_tc, true_tc = pred, target
        return {"pred_tc": pred_tc, "true_tc": true_tc}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]


# =============================================================================
# 4. 主训练流程
# =============================================================================

def build_model(cfg: Dict[str, Any]) -> M3GNet:
    """构建 M3GNet 模型 (可选从预训练加载)"""
    model_cfg = cfg["model"]

    model = M3GNet(
        element_types=DEFAULT_ELEMENTS,
        cutoff=model_cfg.get("cutoff", 5.0),
        threebody_cutoff=model_cfg.get("threebody_cutoff", 4.0),
        dim_node_embedding=model_cfg.get("dim_node_embedding", 64),
        dim_edge_embedding=model_cfg.get("dim_edge_embedding", 64),
        units=model_cfg.get("units", 64),
        nlayers=model_cfg.get("nlayers", 3),
        is_intensive=model_cfg.get("is_intensive", True),
        readout_type="weighted_atom",   # 用加权原子读出
        ntypes_state=None,              # 不用 state features
    )

    # 可选: 从预训练 weights 加载
    pretrained = model_cfg.get("pretrained", None)
    if pretrained:
        print(f"  从预训练模型 {pretrained} 加载权重...")
        try:
            pretrained_model = matgl.load_model(pretrained)
            # 复制 backbone 权重 (跳过最后的 readout 层, 因为 Tc != E_form)
            pretrained_state = pretrained_model.state_dict()
            current_state = model.state_dict()
            transferred = 0
            for k, v in pretrained_state.items():
                if k in current_state and current_state[k].shape == v.shape:
                    current_state[k] = v
                    transferred += 1
            model.load_state_dict(current_state)
            print(f"  ✓ 迁移了 {transferred}/{len(current_state)} 个参数")
        except Exception as e:
            print(f"  ⚠ 预训练加载失败 ({e}), 从头训练")

    return model


def main(config_path: Path):
    print("=" * 70)
    print("  Training M3GNet for Tc Regression")
    print("=" * 70)

    cfg = load_config(config_path)
    print(f"\n配置: {config_path}")
    print(json.dumps(cfg, indent=2, default=str))

    # 设置随机种子
    L.seed_everything(cfg["output"]["seed"])

    # ---- 数据 ----
    print("\n[1/4] 加载数据集...")
    train_records = load_pkl(ROOT / cfg["data"]["train_pkl"])
    val_records = load_pkl(ROOT / cfg["data"]["val_pkl"])
    test_records = load_pkl(ROOT / cfg["data"]["test_pkl"])
    print(f"      train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    converter = Structure2Graph(
        element_types=DEFAULT_ELEMENTS,
        cutoff=cfg["model"]["cutoff"],
    )
    log_transform = cfg["data"]["log_transform"]

    train_ds = records_to_dataset(train_records, converter, log_transform)
    val_ds = records_to_dataset(val_records, converter, log_transform)
    test_ds = records_to_dataset(test_records, converter, log_transform)

    collate = partial(collate_fn_graph, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        collate_fn=collate,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    # ---- 模型 ----
    print("\n[2/4] 构建模型...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {n_params:,}")

    lit_module = TcRegressor(
        model=model,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        log_transform=log_transform,
        loss_type=cfg["training"]["loss"],
    )

    # ---- Trainer ----
    print("\n[3/4] 启动 Trainer...")
    ckpt_dir = ROOT / cfg["output"]["ckpt_dir"]
    log_dir = ROOT / cfg["output"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="m3gnet_tc_{epoch:03d}_{val_tc_mae_K:.2f}",
            monitor="val_tc_mae_K",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_tc_mae_K",
            mode="min",
            patience=cfg["training"]["early_stopping_patience"],
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = CSVLogger(save_dir=str(log_dir), name="csv")

    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # ---- 训练 ----
    print(f"\n      设备: {trainer.accelerator}, GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU 名: {torch.cuda.get_device_name(0)}")
        print(f"      显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer.fit(lit_module, train_loader, val_loader)

    # ---- 测试 + 保存最终模型 ----
    print("\n[4/4] 在测试集上评估最佳模型...")
    trainer.test(lit_module, test_loader, ckpt_path="best")

    # 保存最佳模型到固定路径
    best_path = ROOT / cfg["output"]["best_model"]
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        import shutil
        shutil.copy(best_ckpt, best_path)
        print(f"      ✓ 最佳模型已保存到 {best_path}")

    print("\n" + "=" * 70)
    print("  训练完成. 下一步: python scripts/04_evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_config.yaml",
        help="训练配置 YAML 路径",
    )
    args = parser.parse_args()
    main(args.config)
