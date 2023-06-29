import json
from json import JSONDecodeError
from warnings import warn

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from scripts.examples._sonar_data import get_random_datum_pair
from sips.data import CameraParams, CameraPose, SonarBatch, SonarDatum, SonarDatumPair


class SonarDataset(Dataset[SonarDatumPair]):
    def __init__(self, config, mask=[]) -> None:
        super().__init__()
        self.config = config
        try:
            with open("data/tuples.json") as f:
                tuples = json.load(f)
        except IOError:
            warn(f"tuples.json file missing")
            return
        except JSONDecodeError:
            warn(f"tuples.json file empty")
            return
        try:
            with open("data/poses.json") as f:
                poses = json.load(f)
        except IOError:
            warn(f"poses.json file missing")
            return
        except JSONDecodeError:
            warn(f"poses.json file empty")
            return
        self.poses = poses
        if len(mask) > 0:
            self.tuples = [
                tuple_ for idx, tuple_ in enumerate(tuples) if idx in mask.indices
            ]
        else:
            self.tuples = tuples
        return

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, index) -> SonarDatumPair:
        this_tuple = self.tuples[index]
        pose_data0 = self.poses[str(this_tuple[0])]
        pose_data1 = self.poses[str(this_tuple[1])]
        cam_params = CameraParams(
            min_range=self.config.min_range,
            max_range=self.config.max_range,
            azimuth=self.config.horizontal_fov,
            elevation=self.config.vertical_fov,
        )
        transform = T.Compose([T.PILToTensor()])
        im_path0 = (
            f"data/filtered/{pose_data0['rosbag']}/images/sonar_{this_tuple[0]}.png"
        )
        im_path1 = (
            f"data/filtered/{pose_data1['rosbag']}/images/sonar_{this_tuple[1]}.png"
        )
        try:
            im0 = Image.open(im_path0)
        except FileNotFoundError:
            warn("Image file not found, skipping the rest of the operations")
            return None  # type: ignore[return-value]
        im_tensor0 = transform((im0.resize(self.config.image_shape)))
        try:
            im1 = Image.open(im_path1)
        except FileNotFoundError:
            warn("Image file not found, skipping the rest of the operations")
            return None  # type: ignore[return-value]
        im_tensor1 = transform((im1.resize(self.config.image_shape)))
        pose0 = CameraPose(
            position=[
                pose_data0["point_position"]["x"],
                pose_data0["point_position"]["y"],
                pose_data0["point_position"]["z"],
            ],
            rotation=[
                pose_data0["quaterion_orientation"]["x"],
                pose_data0["quaterion_orientation"]["y"],
                pose_data0["quaterion_orientation"]["z"],
                pose_data0["quaterion_orientation"]["w"],
            ],
        )
        pose1 = CameraPose(
            position=[
                pose_data1["point_position"]["x"],
                pose_data1["point_position"]["y"],
                pose_data1["point_position"]["z"],
            ],
            rotation=[
                pose_data1["quaterion_orientation"]["x"],
                pose_data1["quaterion_orientation"]["y"],
                pose_data1["quaterion_orientation"]["z"],
                pose_data1["quaterion_orientation"]["w"],
            ],
        )
        return SonarDatumPair(
            SonarDatum(image=im_tensor0, pose=pose0, params=cam_params),
            SonarDatum(image=im_tensor1, pose=pose1, params=cam_params),
        )


class DummySonarDataSet(SonarDataset):
    def __init__(self, n: int = 100) -> None:
        warn("WARNING: You are using a dummy dataset!")
        super().__init__()  # type: ignore[call-arg]
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> SonarDatumPair:
        return get_random_datum_pair()
