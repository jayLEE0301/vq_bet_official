import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from vqvae.vqvae import *
import wandb

config_name = "pretrain_ant"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)

    vqvae_model = hydra.utils.instantiate(cfg.vqvae_model)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    for epoch in tqdm.trange(cfg.epochs):
        for data in tqdm.tqdm(train_loader):
            obs, act, goal = (x.to(cfg.device) for x in data)

            (
                encoder_loss,
                vq_loss_state,
                vq_code,
                vqvae_recon_loss,
            ) = vqvae_model.vqvae_update(act)  # N T D

            wandb.log({"pretrain/n_different_codes": len(torch.unique(vq_code))})
            wandb.log(
                {"pretrain/n_different_combinations": len(torch.unique(vq_code, dim=0))}
            )
            wandb.log({"pretrain/encoder_loss": encoder_loss})
            wandb.log({"pretrain/vq_loss_state": vq_loss_state})
            wandb.log({"pretrain/vqvae_recon_loss": vqvae_recon_loss})

        if epoch % 50 == 0:
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))


if __name__ == "__main__":
    main()
