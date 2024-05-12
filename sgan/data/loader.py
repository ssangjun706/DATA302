from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    # dset은 데이터를 뿌려줄 때 (*, 2, 16) 형태로 뿌려주고, *는 얼만지 알 수 없음 (개별 Sequnce마다 다르므로)
    # len(dset)은 총 Sequence의 수로, 전체 데이터 수와는 다름.

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,                 # Batch 크기, 기본 = 64
        shuffle=True,
        num_workers=args.loader_num_workers,        # 데이터를 GPU에 로드할 때 사용할 스레드 수
        collate_fn=seq_collate)                     # 부족한 부분을 패딩하는 함수
    return dset, loader
