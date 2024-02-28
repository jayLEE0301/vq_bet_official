import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import einops
import numpy as np
import torch
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset, Subset, TensorDataset
import pathlib
import tqdm
import struct
import torch.nn.functional as F


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


import zarr


class PushTSequenceDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self, data_directory, device="cuda", onehot_goals=False, visual_input=False
    ):
        data_directory = Path(data_directory)
        src_root = zarr.group(data_directory / "pusht_cchi_v7_replay.zarr")
        self.visual_input = visual_input
        # numpy backend
        meta = dict()
        for key, value in src_root["meta"].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]

        keys = src_root["data"].keys()
        data = dict()
        for key in keys:
            arr = src_root["data"][key]
            data[key] = arr[:]

        if not self.visual_input:
            observations = []
        actions = []
        masks = []
        goals = []
        start = 0
        agent_pos = data["state"][:, :2]
        keypoint_obs = np.concatenate(
            [data["keypoint"].reshape(data["keypoint"].shape[0], -1), agent_pos],
            axis=-1,
        )

        for end in meta["episode_ends"]:
            if (300 - (end - start)) <= 0:
                print("too small capacity")
            if not self.visual_input:
                observations.append(
                    np.concatenate(
                        (keypoint_obs[start:end], np.zeros((300 - (end - start), 20)))
                    )
                )
            actions.append(
                np.concatenate(
                    (data["action"][start:end], np.zeros((300 - (end - start), 2)))
                )
            )
            masks.append(
                np.concatenate(
                    (np.ones((end - start)), np.zeros((300 - (end - start))))
                )
            )
            goals.append(
                np.concatenate(
                    (data["state"][start:end, 2:], np.zeros((300 - (end - start), 3)))
                )
            )

            start = end
        if not self.visual_input:
            observations = np.array(observations)
        actions = np.array(actions)
        masks = np.array(masks)
        goals = np.array(goals)
        if not self.visual_input:
            self.observations_stats = self.get_data_stats(observations)
            observations = self.normalize_data(observations, self.observations_stats)
            observations = torch.from_numpy(observations)[:].float()
        actions = torch.from_numpy(actions)[:].float()
        goals = torch.from_numpy(goals)[:].float()
        goals = torch.concat(
            (goals, torch.zeros((goals.shape[0], goals.shape[1], 17))), dim=2
        )
        masks = torch.from_numpy(masks)[:].float()

        if visual_input:
            observations = []
            for i in tqdm.trange(206):
                img_obs_epi = torch.load(
                    data_directory
                    / "image_resnet_embedding"
                    / "demostensor_epi_{}.pth".format(i),
                    map_location=torch.device("cpu"),
                )
                img_obs_epi.requires_grad_(False)
                observations.append(img_obs_epi.to(device))
            observations = torch.stack(observations)

        self.masks = masks
        tensors = [observations, actions]

        tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_data_stats(self, data):
        data = data.reshape(-1, data.shape[-1])
        stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
        return stats

    def normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats["max"] - stats["min"]) + stats["min"]
        return data

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])


class RelayKitchenTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self, data_directory, device="cuda", onehot_goals=False, visual_input=False
    ):
        data_directory = Path(data_directory)
        if visual_input:
            observations = []
            for i in tqdm.trange(566):
                img_obs_epi = torch.load(
                    data_directory
                    / "image_resnet_embedding"
                    / "demostensor_epi_{}.pth".format(i),
                    map_location=torch.device("cpu"),
                )
                img_obs_epi.requires_grad_(False)
                observations.append(img_obs_epi.to(device))
            observations = torch.stack(observations)
        else:
            observations = torch.from_numpy(
                np.load(data_directory / "observations_seq.npy")
            ).to(device)
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy")).to(
            device
        )
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy")).to(
            device
        )
        goals = torch.load(data_directory / "onehot_goals.pth").to(device)
        # The current values are in shape T x N x Dim, move to N x T x Dim
        if visual_input:
            actions, masks, goals = transpose_batch_timestep(actions, masks, goals)
        else:
            observations, actions, masks, goals = transpose_batch_timestep(
                observations, actions, masks, goals
            )
        print("data size:", observations.shape, actions.shape)

        self.masks = masks
        tensors = [observations, actions]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class AntTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cuda", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = torch.cat(
            (
                torch.from_numpy(np.load(data_directory / "ob_save.npy")),
                torch.from_numpy(np.load(data_directory / "goal_save.npy")),
            ),
            -1,
        ).cuda()
        actions = torch.from_numpy(np.load(data_directory / "a_save.npy")).cuda()
        masks = torch.from_numpy(np.load(data_directory / "mask_save.npy")).cuda()
        goals = torch.from_numpy(np.load(data_directory / "goal_save.npy")).cuda()

        self.masks = masks
        tensors = [observations, actions]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class UR3TrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self, data_directory, device="cuda", onehot_goals=False, seed=0, n_episode=600
    ):
        data_directory = Path(data_directory)
        observations = torch.from_numpy(np.load(data_directory / "data_obs.npy")).cuda()
        actions = torch.from_numpy(np.load(data_directory / "data_act.npy")).cuda()
        masks = torch.from_numpy(np.load(data_directory / "data_msk.npy")).cuda()
        goals = torch.from_numpy(np.load(data_directory / "data_obs.npy")).cuda()
        # goals = torch.from_numpy(np.load(data_directory / "goal_save.npy")).cuda()
        # The current values are in shape T x N x Dim, move to N x T x Dim
        # observations, actions, masks, goals = transpose_batch_timestep(
        #     observations, actions, masks, goals
        # )
        self.masks = masks
        tensors = [observations, actions]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class PushTrajectorySequenceDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cuda", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = (
            torch.from_numpy(
                np.load(data_directory / "multimodal_push_observations.npy")
            )
            .cuda()
            .float()
        )
        actions = (
            torch.from_numpy(np.load(data_directory / "multimodal_push_actions.npy"))
            .cuda()
            .float()
        )
        masks = (
            torch.from_numpy(np.load(data_directory / "multimodal_push_masks.npy"))
            .cuda()
            .float()
        )
        goals = torch.load(data_directory / "onehot_goals.pth").cuda().float()
        # The current values are in shape T x N x Dim, move to N x T x Dim
        # observations, actions, masks, goals = transpose_batch_timestep(
        #     observations, actions, masks, goals
        # )
        self.masks = masks
        tensors = [observations, actions]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = True,
        future_conditional: bool = False,
        get_goal_from_dataset: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.get_goal_from_dataset = get_goal_from_dataset
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        if vqbet_get_future_action_chunk:
            min_window_required = window + action_window
        else:
            min_window_required = max(window, action_window)
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - min_window_required < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={min_window_required}"
                )
            else:
                self.slices += [
                    (i, 0, end + 1) for end in range(window - 1)
                ]  # slice indices follow convention [start, end)
                self.slices += [
                    (i, start, start + window)
                    for start in range(T - min_window_required)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < min_window_required:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        if self.vqbet_get_future_action_chunk:
            if end - start < self.window:
                if self.dataset[i][0].ndim > 2:
                    values = [
                        torch.cat(
                            (
                                torch.tile(
                                    self.dataset[i][0][start],
                                    ((self.window - (end - start)), 1, 1, 1),
                                ),
                                self.dataset[i][0][start:end],
                            ),
                            dim=0,
                        ),
                        torch.cat(
                            (
                                torch.tile(
                                    self.dataset[i][1][start],
                                    ((self.window - (end - start)), 1),
                                ),
                                self.dataset[i][1][
                                    start : end - 1 + self.action_window
                                ],
                            ),
                            dim=-2,
                        ),
                    ]
                else:
                    values = [
                        torch.cat(
                            (
                                torch.tile(
                                    self.dataset[i][0][start],
                                    ((self.window - (end - start)), 1),
                                ),
                                self.dataset[i][0][start:end],
                            ),
                            dim=-2,
                        ),
                        torch.cat(
                            (
                                torch.tile(
                                    self.dataset[i][1][start],
                                    ((self.window - (end - start)), 1),
                                ),
                                self.dataset[i][1][
                                    start : end - 1 + self.action_window
                                ],
                            ),
                            dim=-2,
                        ),
                    ]
            else:
                values = [
                    self.dataset[i][0][start:end],
                    self.dataset[i][1][start : end - 1 + self.action_window],
                ]
        else:
            if end - start < self.window:
                values = [
                    torch.unsqueeze(self.dataset[i][0][start], dim=0),
                    self.dataset[i][1][start : start + self.action_window],
                ]
            else:
                values = [
                    torch.unsqueeze(self.dataset[i][0][start], dim=0),
                    self.dataset[i][1][start : start + self.action_window],
                ]
        if self.get_goal_from_dataset:
            valid_start_range = (
                end + self.min_future_sep,
                self.dataset.get_seq_length(i) - self.future_seq_len,
            )
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    future_obs = self.dataset[i][2][-self.future_seq_len :]
                else:
                    start = np.random.randint(*valid_start_range)
                    end = start + self.future_seq_len
                    future_obs = self.dataset[i][2][start:end]
            else:
                future_obs = self.dataset[i][2][-self.future_seq_len :]

            # [observations, actions, mask[, future_obs (goal conditional)]]
            values.append(future_obs)

        elif self.future_conditional:
            valid_start_range = (
                end + self.min_future_sep,
                self.dataset.get_seq_length(i) - self.future_seq_len,
            )
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    future_obs = self.dataset[i][0][-self.future_seq_len :]
                else:
                    start = np.random.randint(*valid_start_range)
                    end = start + self.future_seq_len
                    future_obs = self.dataset[i][0][start:end]
            else:
                # if image-based data
                if self.dataset[i][0].ndim > 2:
                    future_obs = self.dataset[i][0][-self.future_seq_len :]
                # if state-based data
                else:
                    # zeros placeholder T x obs_dim
                    _, obs_dim = values[0].shape
                    future_obs = torch.zeros((self.future_seq_len, obs_dim)).cuda()

            # [observations, actions, mask[, future_obs (goal conditional)]]
            values.append(future_obs)

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    window_size: int = 10,
    action_window_size: int = 10,
    vqbet_get_future_action_chunk: bool = True,
    future_conditional: bool = False,
    get_goal_from_dataset: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "action_window": action_window_size,
        "vqbet_get_future_action_chunk": vqbet_get_future_action_chunk,
        "future_conditional": future_conditional,
        "get_goal_from_dataset": get_goal_from_dataset,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
    }
    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)

    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_pusht_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    window_size=10,
    action_window_size=10,
    vqbet_get_future_action_chunk: bool = True,
    only_sample_tail: bool = False,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
    visual_input=False,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        PushTSequenceDataset(
            data_directory,
            onehot_goals=(goal_conditional == "onehot"),
            visual_input=visual_input,
            device="cuda",
        ),
        train_fraction,
        random_seed,
        window_size,
        action_window_size,
        vqbet_get_future_action_chunk,
        only_sample_tail=only_sample_tail,
        future_conditional=(goal_conditional == "future"),
        get_goal_from_dataset=(not visual_input),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    window_size=10,
    action_window_size=10,
    vqbet_get_future_action_chunk: bool = True,
    only_sample_tail: bool = False,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
    visual_input=False,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        RelayKitchenTrajectoryDataset(
            data_directory,
            onehot_goals=(goal_conditional == "onehot"),
            visual_input=visual_input,
        ),
        train_fraction,
        random_seed,
        window_size,
        action_window_size,
        vqbet_get_future_action_chunk,
        only_sample_tail=only_sample_tail,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def get_ant_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    window_size=10,
    action_window_size=10,
    vqbet_get_future_action_chunk: bool = True,
    only_sample_tail: bool = False,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        AntTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        window_size,
        action_window_size,
        vqbet_get_future_action_chunk,
        only_sample_tail=only_sample_tail,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def get_ur3_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    window_size=10,
    action_window_size=10,
    vqbet_get_future_action_chunk: bool = True,
    only_sample_tail: bool = False,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        UR3TrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        window_size,
        action_window_size,
        vqbet_get_future_action_chunk,
        only_sample_tail=only_sample_tail,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)
