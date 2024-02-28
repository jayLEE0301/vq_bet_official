import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import kitchen_env
import wandb
from video import VideoRecorder
import pickle

config_name = "train_ant_goalcond"

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=False
    )
    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )
    env = hydra.utils.instantiate(cfg.env.gym)
    goal_fn = hydra.utils.instantiate(cfg.goal_fn)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @torch.no_grad()
    def eval_on_env(
        cfg,
        num_evals=cfg.num_env_evals,
        num_eval_per_goal=1,
        videorecorder=None,
        epoch=None,
    ):
        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []  # only used in pusht env
        avg_final_coverage = []  # only used in pusht env
        for goal_idx in range(num_evals):
            if videorecorder is not None:
                videorecorder.init(enabled=(goal_idx == 0))
            for _ in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)
                obs_stack.append(env.reset())
                done, step, total_reward = False, 0, 0
                goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                while not done:
                    obs = torch.from_numpy(np.stack(obs_stack)).float().to(cfg.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                    action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)
                    if cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    obs, reward, done, info = env.step(curr_action)
                    if videorecorder.enabled:
                        videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    obs_stack.append(obs)
                    if "pusht" not in config_name:
                        goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                avg_reward += total_reward
                if "pusht" in config_name:
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info["final_coverage"])
                completion_id_list.append(info["all_completions_ids"])
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()
        if (epoch % cfg.eval_on_env_freq == 0):
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                cfg,
                videorecorder=video,
                epoch=epoch,
                num_eval_per_goal=cfg.num_final_eval_per_goal,
            )
            with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
                pickle.dump(completion_id_list, fp)
            wandb.log({"eval_on_env": avg_reward})
            if "pusht" in config_name:
                wandb.log(
                    {"final coverage mean": sum(final_coverage) / len(final_coverage)}
                )
                wandb.log({"final coverage max": max(final_coverage)})
                wandb.log({"final coverage min": min(final_coverage)})
                wandb.log({"max coverage mean": sum(max_coverage) / len(max_coverage)})
                wandb.log({"max coverage max": max(max_coverage)})
                wandb.log({"max coverage min": min(max_coverage)})

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device) for x in data)
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}")
            wandb.log({"eval/epoch_wise_action_diff": action_diff})
            wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        for data in tqdm.tqdm(train_loader):
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].zero_grad()
                optimizer["optimizer2"].zero_grad()
            else:
                optimizer["optimizer2"].zero_grad()
            obs, act, goal = (x.to(cfg.device) for x in data)
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            loss.backward()
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].step()
                optimizer["optimizer2"].step()
            else:
                optimizer["optimizer2"].step()

        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)

    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        num_eval_per_goal=cfg.num_final_eval_per_goal,
        videorecorder=video,
        epoch=cfg.epochs,
    )
    if "pusht" in config_name:
        wandb.log({"final coverage mean": sum(final_coverage) / len(final_coverage)})
        wandb.log({"final coverage max": max(final_coverage)})
        wandb.log({"final coverage min": min(final_coverage)})
        wandb.log({"max coverage mean": sum(max_coverage) / len(max_coverage)})
        wandb.log({"max coverage max": max(max_coverage)})
        wandb.log({"max coverage min": min(max_coverage)})
    with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    wandb.log({"final_eval_on_env": avg_reward})
    return avg_reward


if __name__ == "__main__":
    main()
