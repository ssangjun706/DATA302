import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    # 64개의 Sequence 데이터
    # obs_seq_list는 (64, *, 2, 8) 형태
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    # sequence마다 보행자 수가 다르므로 리스트화
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()

    # 이번에도 시작, 끝을 배열로 두어서 Seq 내 관계성을 유지시켜 줄 듯.
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size

    # 마찬가지로, 데이터를 다시 (64*(Ped per Seq), 2, 8) 형태로 뭉침
    # 그리고, (Frame, 64*(Ped Per Seq), 좌표) 형태로 바꿈 ----> Frame 단위로 Training을 할 수 있도록
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    # 마찬가지로, Seq 별 관계를 깨트리지 않기 위해 순서 데이터도 첨부함.
    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = "\t"
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            line = [float(i) for i in line]     # [0.0, 1.0, 1.4, -5.74]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        # timestep: LSTM의 각 단계 (즉, timestep=8이면 8개의 hidden state를 갖는 LSTM)

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir            # dataset/~~~/~~~.txt
        self.obs_len = obs_len              # Number of time-steps in input trajectories, default: 8
        self.pred_len = pred_len            # Number of time-steps in output trajectories, default: 8
        self.skip = skip                    # 1
        self.seq_len = self.obs_len + self.pred_len     # seq_len -> 한 번에 이용할 행 수
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            # data[:, 0] -> N개의 행에서 첫 번째 (Frame)만 추출 (N, 1)
            # np.unique() -> 중복되는 원소는 제거함

            frame_data = []
            # idx == data[:,0]  ->  idx를 [N, 1] 배열로 늘려서 element-wise 비교 실행

            for frame in frames:    # 예시: frames -> [0, 10, ..., 14390]
                frame_data.append(data[frame == data[:, 0], :])
                # data 배열에서 첫 번째 원소가 frame에 해당하는 것만 추출

            # frame_data -> [[... frame 1 ...], [... frame 2 ...], ... , [... frame N ... ]]
            # frame_data.shape == (total frames, index per frame, 4)

            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            # num_sequence: 데이터를 Sequence 단위로 끊었을 때 마지막에 남는 것을 처리하기 위한 것으로 추측
            # ex) 데이터 수가 934개이고, Sequence 단위가 16이면, 마지막 Sequnce는 데이터가 부족
            # 934 -> 919

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                # 16개의 Frame 묶음을 하나의 텐서로 변환 -> (M, 4)
                # ex) 0~15 frame까지 데이터를 한 묶음으로 변환

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # 한 묶음으로 변환한 거에서 index를 추출 해 냄 -> 보행자 추출

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # ex) 보행자가 13명이면 (13, 2, 16) 크기의 zero-tensor를 생성함

                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # curr_seq_rel과 같은 크기

                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                # ex) 보행자가 13명이면 (13, 16) 크기 배열을 생성

                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):   # 모든 보행자에 대해서 탐색할 것
                    # curr_ped_seq -> curr_seq_data(묶음 데이터 [N, 4])에서 보행자 i인 데이터만을 추출
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # ped_id 보행자가 등장한 첫 번째 프레임
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # curr_ped_seq[0, 0]: ped_id 번째 보행자 데이터 중 가장 첫 번째 프레임
                    # frames.index(): frame에서 위치 -> 비어있는 frame이 존재할 수 있음

                    # ped_id 보행자가 등장한 마지막 프레임
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue

                    # 보행자 데이터 개수가 seq_len(16)인 것만 이용할 것임
                    # 0번부터 N-seq_len번 프레임까지 한 프레임씩 이동하면서
                    # 각 단계에 해당하는 범위(16개) 내 모든 프레임에서 이동 흔적이 있는 보행자만 이용한다는 뜻.

                    # frame, id 정보를 삭제.
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])

                    # ped_id 보행자의 이동을 배열로 나타내도록 함 -> (2, 16) 크기 -> (x, y) 좌표가 16개 쌓인 형태
                    curr_ped_seq = curr_ped_seq

                    # 상대좌표로 변환시키는 작업
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)

                    # 1~N까지 데이터에서 0~(N-1)까지 데이터를 원소끼리 뺄셈.
                    # 1~N까지 데이터에 뺄셈 결과를 덮어쓰기함.
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    _idx = num_peds_considered      # 초기값 0
                    # curr_seq -> (13, 2, 16) 크기의 0 배열
                    # curr_ped_seq는 (2, M) 크기의 배열

                    # 16개인 보행자 데이터만 0~num_ped_considered까지 채워넣고
                    # 나머지 보행자 데이터는 0벡터로 초기화
                    # 16개 모두 존재하는 보행자 데이터이므로 pad_front=0, pad_end=self.seq_len+1임.
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                # Sequence에 속한 보행자 중 데이터 수가 seq_len인 것들만 골라내고
                # 그 보행자 수가 1명 이상인 Sequence
                if num_peds_considered > min_ped:       # min_ped: 1
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])

                    # 0~num_peds_considered 까지만 유효 데이터이므로 나머지 데이터는 버림
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        # seq_list: (유효 Sequence 수, (유효 보행자 수, (좌표, Frame)))
        # 유효 보행자는 Sequence의 모든 Frame에 데이터가 존재하는 경우
        # 유효 Sequence는 Sequence 내 유효 보행자 수가 1 이상
        # 시퀀스 단위로 재구성했으므로, 원래 프레임에서의 순서는 크게 중요하지 않음

        self.num_seq = len(seq_list)

        # Sequence 구분도 무의미하므로 유효 보행자의 개별 Frame 데이터만 남긴채 뭉개버림
        # (유효 seq 수 * 유효 보행자 수, 좌표, frame) 형태로 변환
        seq_list = np.concatenate(seq_list, axis=0) # dim=4 Tensor
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # 유효 보행자 경로 데이터를 전부 불러오고
        # 각 보행자마다 Frame을 전반부(관측)와 후반부(예측)로 분리
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        # num_peds_in_seq : 각 영상에서 각 Seq마다 보행자 데이터를 정제했고
        # 개별 Seq에서 몇 명의 보행자를 추출했는지 숫자로 표시한 걸 리스트에 누적해 놓음
        # 그니까, 개별 Sequence간 관계는 없어졌지만 한 Sequence 안에서의 보행자 정보의 관계는 유지하겠다 이거네.
        # 그래서 Social이라는게 붙었구나.

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # 누적합으로 변환하여 데이터 처리하기 쉽게 바꿈

        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        # 데이터를 Sequence 단위로 추출하도록 만들었음

        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
