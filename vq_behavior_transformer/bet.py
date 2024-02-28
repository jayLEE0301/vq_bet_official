import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from vq_behavior_transformer.gpt import GPT
from vq_behavior_transformer.utils import MLP
from vqvae.vqvae import VqVae


class BehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        gpt_model: GPT,
        vqvae_model: VqVae,
        offset_loss_multiplier: float = 1.0e3,
        secondary_code_multiplier: float = 0.5,
        gamma: float = 2.0,
        obs_window_size=10,
        act_window_size=10,
        sequentially_select=False,
        visual_input=False,
        finetune_resnet=False,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._goal_dim = goal_dim
        self.obs_window_size = obs_window_size
        self.act_window_size = act_window_size
        self.sequentially_select = sequentially_select
        if goal_dim <= 0:
            self._cbet_method = self.GOAL_SPEC.unconditional
        elif obs_dim == goal_dim:
            self._cbet_method = self.GOAL_SPEC.concat
        else:
            self._cbet_method = self.GOAL_SPEC.stack
        print("initialize VQ-BeT agent")

        self._gpt_model = gpt_model
        self._vqvae_model = vqvae_model
        # For now, we assume the number of clusters is given.

        self._G = self._vqvae_model.vqvae_groups  # G(number of groups)
        self._C = self._vqvae_model.vqvae_n_embed  # C(number of code integers)
        self._D = self._vqvae_model.embedding_dim  # D(embedding dims)

        if self.sequentially_select:
            print("use sequantial prediction for vq dictionary!")
            self._map_to_cbet_preds_bin1 = MLP(
                in_channels=gpt_model.config.output_dim,
                hidden_channels=[512, 512, self._C],
            )
            self._map_to_cbet_preds_bin2 = MLP(
                in_channels=gpt_model.config.output_dim + self._C,
                hidden_channels=[512, self._C],
            )
        else:
            self._map_to_cbet_preds_bin = MLP(
                in_channels=gpt_model.config.output_dim,
                hidden_channels=[1024, 1024, self._G * self._C],
            )
        self._map_to_cbet_preds_offset = MLP(
            in_channels=gpt_model.config.output_dim,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (act_dim * self.act_window_size),
            ],
        )

        if visual_input:
            self._resnet_header = MLP(
                in_channels=512,
                hidden_channels=[1024],
            )
        self._collected_actions = []
        self._have_fit_kmeans = False
        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        # Placeholder for the cluster centers.
        self._criterion = FocalLoss(gamma=gamma)
        self.visual_input = visual_input
        self.finetune_resnet = finetune_resnet
        if visual_input:
            import torchvision.models as models
            import torchvision.transforms as transforms

            resnet = models.resnet18(pretrained=True)
            self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).cuda()
            if not self.finetune_resnet:
                for param in self.resnet.parameters():
                    param.requires_grad = False
            self.transform = transforms.Compose(
                [  # transforms.Resize((224, 224)), \
                    # if error -> delete Resize and add F.interpolate before applying self.transform.
                    # TODO np to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]
                    )
                ]
            )

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._predict(obs_seq, goal_seq, action_seq)

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        if self.visual_input:
            obs_seq = obs_seq.cuda()
            if obs_seq.ndim == 3:
                obs_seq = obs_seq.clone().detach()
                obs_seq = self._resnet_header(obs_seq)
            else:
                N = obs_seq.shape[0]
                if obs_seq.shape[-1] == 3:
                    obs_seq = (
                        einops.rearrange(obs_seq, "N T W H C -> (N T) C W H") / 255.0
                    )  # * 2. - 1.
                else:
                    obs_seq = (
                        einops.rearrange(obs_seq, "N T C W H -> (N T) C W H") / 255.0
                    )  # * 2. - 1.
                # obs_seq = obs_seq.cuda()
                if obs_seq.shape[-1] != 224:
                    obs_seq = F.interpolate(obs_seq, size=224)
                obs_seq = self.transform(obs_seq)
                obs_seq = torch.squeeze(torch.squeeze(self.resnet(obs_seq), -1), -1)
                obs_seq = self._resnet_header(obs_seq)
                obs_seq = einops.rearrange(obs_seq, "(N T) L -> N T L", N=N)
            if not (self._cbet_method == self.GOAL_SPEC.unconditional):
                goal_seq = goal_seq.cuda()
                if goal_seq.ndim == 3:
                    goal_seq = goal_seq.clone().detach()
                    goal_seq = self._resnet_header(goal_seq)
                else:
                    if goal_seq.shape[-1] == 3:
                        goal_seq = (
                            einops.rearrange(goal_seq, "N T W H C -> (N T) C W H")
                            / 255.0
                        )  # * 2. - 1.
                    else:
                        goal_seq = (
                            einops.rearrange(goal_seq, "N T C W H -> (N T) C W H")
                            / 255.0
                        )  # * 2. - 1.
                    # goal_seq = goal_seq.cuda()
                    if goal_seq.shape[-1] != 224:
                        goal_seq = F.interpolate(goal_seq, size=224)
                    goal_seq = self.transform(goal_seq)
                    goal_seq = torch.squeeze(
                        torch.squeeze(self.resnet(goal_seq), -1), -1
                    )
                    goal_seq = self._resnet_header(goal_seq)
                    goal_seq = einops.rearrange(goal_seq, "(N T) L -> N T L", N=N)
        if obs_seq.shape[1] < self.obs_window_size:
            obs_seq = torch.cat(
                (
                    torch.tile(
                        obs_seq[:, 0, :],
                        (1, self.obs_window_size - obs_seq.shape[1], 1),
                    ),
                    obs_seq,
                ),
                dim=-2,
            )
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_input = obs_seq
        elif self._cbet_method == self.GOAL_SPEC.concat:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=1)
        elif self._cbet_method == self.GOAL_SPEC.stack:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)
        else:
            raise NotImplementedError
        gpt_output = self._gpt_model(gpt_input)

        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_output = gpt_output
        else:
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        gpt_output = einops.rearrange(gpt_output, "N T (G C) -> (N T) (G C)", G=self._G)
        obs = einops.rearrange(obs_seq, "N T O -> (N T) O")
        obs = obs.unsqueeze(dim=1)

        if self.sequentially_select:
            cbet_logits1 = self._map_to_cbet_preds_bin1(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs1 = torch.softmax(cbet_logits1, dim=-1)
            NT, choices = cbet_probs1.shape
            G = self._G
            sampled_centers1 = einops.rearrange(
                torch.multinomial(cbet_probs1.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(sampled_centers1, num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_probs2 = torch.softmax(cbet_logits2, dim=-1)
            sampled_centers2 = einops.rearrange(
                torch.multinomial(cbet_probs2.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack(
                (sampled_centers1, sampled_centers2), axis=1
            )  # NT, G
        else:
            cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
            )
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs = torch.softmax(cbet_logits, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        indices = (
            torch.arange(NT).unsqueeze(1).cuda(),
            torch.arange(self._G).unsqueeze(0).cuda(),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]  # NT, G, W, A(?) or NT, G, A

        sampled_offsets = sampled_offsets.sum(dim=1)
        centers = self._vqvae_model.draw_code_forward(sampled_centers).view(
            NT, -1, self._D
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT G D -> NT (G D)"
        )
        decoded_action = (
            self._vqvae_model.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )  # NT, A
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self._vqvae_model.input_dim_h
        )
        predicted_action = decoded_action + sampled_offsets
        if action_seq is not None:
            n, total_w, act_dim = action_seq.shape
            act_w = self._vqvae_model.input_dim_h
            obs_w = total_w + 1 - act_w
            output_shape = (n, obs_w, act_w, act_dim)
            output = torch.empty(output_shape).to(action_seq.device)
            for i in range(obs_w):
                output[:, i, :, :] = action_seq[:, i : i + act_w, :]
            action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
            # Figure out the loss for the actions.
            # First, we need to find the closest cluster center for each action.
            state_vq, action_bins = self._vqvae_model.get_code(
                action_seq
            )  # action_bins: NT, G

            # Now we can compute the loss.
            if action_seq.ndim == 2:
                action_seq = action_seq.unsqueeze(0)

            offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

            action_diff = F.mse_loss(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ],
                einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ],
            )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
            action_diff_tot = F.mse_loss(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, :, :
                ],
                einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, :, :
                ],
            )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
            action_diff_mean_res1 = (
                abs(
                    einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                        :, -1, 0, :
                    ]
                    - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
                        :, -1, 0, :
                    ]
                )
            ).mean()
            action_diff_mean_res2 = (
                abs(
                    einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                        :, -1, 0, :
                    ]
                    - einops.rearrange(
                        predicted_action, "(N T) W A -> N T W A", T=obs_w
                    )[:, -1, 0, :]
                )
            ).mean()
            action_diff_max = (
                abs(
                    einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                        :, -1, 0, :
                    ]
                    - einops.rearrange(
                        predicted_action, "(N T) W A -> N T W A", T=obs_w
                    )[:, -1, 0, :]
                )
            ).max()

            if self.sequentially_select:
                cbet_loss1 = self._criterion(  # F.cross_entropy
                    cbet_logits1[:, :],
                    action_bins[:, 0],
                )
                cbet_logits2 = self._map_to_cbet_preds_bin2(
                    torch.cat(
                        (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
                        axis=1,
                    )
                )
                cbet_loss2 = self._criterion(  # F.cross_entropy
                    cbet_logits2[:, :],
                    action_bins[:, 1],
                )
            else:
                cbet_loss1 = self._criterion(  # F.cross_entropy
                    cbet_logits[:, 0, :],
                    action_bins[:, 0],
                )
                cbet_loss2 = self._criterion(  # F.cross_entropy
                    cbet_logits[:, 1, :],
                    action_bins[:, 1],
                )
            cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

            equal_total_code_rate = (
                torch.sum(
                    (
                        torch.sum((action_bins == sampled_centers).int(), axis=1) == G
                    ).int()
                )
                / NT
            )
            equal_single_code_rate = torch.sum(
                (action_bins[:, 0] == sampled_centers[:, 0]).int()
            ) / (NT)
            equal_single_code_rate2 = torch.sum(
                (action_bins[:, 1] == sampled_centers[:, 1]).int()
            ) / (NT)

            loss = cbet_loss + self._offset_loss_multiplier * offset_loss
            loss_dict = {
                "classification_loss": cbet_loss.detach().cpu().item(),
                "offset_loss": offset_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "equal_total_code_rate": equal_total_code_rate,
                "equal_single_code_rate": equal_single_code_rate,
                "equal_single_code_rate2": equal_single_code_rate2,
                "action_diff": action_diff.detach().cpu().item(),
                "action_diff_tot": action_diff_tot.detach().cpu().item(),
                "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
                "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
                "action_diff_max": action_diff_max.detach().cpu().item(),
            }
            return predicted_action, loss, loss_dict

        return predicted_action, None, {}

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer1 = self._gpt_model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )

        if self.sequentially_select:
            optimizer1.add_param_group(
                {"params": self._map_to_cbet_preds_bin1.parameters()}
            )
            optimizer1.add_param_group(
                {"params": self._map_to_cbet_preds_bin2.parameters()}
            )
        else:
            optimizer1.add_param_group(
                {"params": self._map_to_cbet_preds_bin.parameters()}
            )
        optimizer2 = torch.optim.AdamW(
            self._map_to_cbet_preds_offset.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        if self.visual_input and self.finetune_resnet:
            optimizer1.add_param_group(
                {"params": self.resnet.parameters(), "lr": learning_rate * 0.1}
            )
        if self.visual_input:
            optimizer1.add_param_group({"params": self._resnet_header.parameters()})
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        return {"optimizer1": optimizer1, "optimizer2": optimizer2}

    def save_model(self, path: Path):
        torch.save(self.state_dict(), path / "cbet_model.pt")
        torch.save(self._gpt_model.state_dict(), path / "gpt_model.pt")
        if hasattr(self, "resnet"):
            torch.save(self.resnet.state_dict(), path / "resnet.pt")
            torch.save(self._resnet_header.state_dict(), path / "resnet_header.pt")
        torch.save(self.optimizer1.state_dict(), path / "optimizer1.pt")
        torch.save(self.optimizer2.state_dict(), path / "optimizer2.pt")

    def load_model(self, path: Path):
        if (path / "cbet_model.pt").exists():
            self.load_state_dict(torch.load(path / "cbet_model.pt"))
        elif (path / "gpt_model.pt").exists():
            self._gpt_model.load_state_dict(torch.load(path / "gpt_model.pt"))
        elif (path / "resnet.pt").exists():
            self.resnet.load_state_dict(torch.load(path / "resnet.pt"))
        elif (path / "optimizer1.pt").exists():
            self.optimizer1.load_state_dict(torch.load(path / "optimizer1.pt"))
        elif (path / "optimizer2.pt").exists():
            self.optimizer2.load_state_dict(torch.load(path / "optimizer2.pt"))
        else:
            logging.warning("No model found at %s", path)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
