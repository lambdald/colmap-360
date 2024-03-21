import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from database import COLMAPDatabase
from rich.progress import track


@dataclass
class ImageFeature:
    image_name: str
    keypoints: np.ndarray
    order: Optional[int] = None

    @staticmethod
    def from_colmap(database_path: Path) -> Dict[int, "ImageFeature"]:
        db = COLMAPDatabase.connect(database_path)
        images_dict = {}  #
        rows = db.execute("SELECT image_id, name FROM images")
        for image_id, image_name in track(rows, description="load images"):
            _, n_kpt, n_kpt_dim, data = next(db.execute(f"SELECT * FROM keypoints WHERE image_id={image_id}"))
            kpts = np.frombuffer(data, np.float32).reshape(n_kpt, n_kpt_dim)
            assert n_kpt_dim in (4, 6)
            if n_kpt_dim == 6:
                kpts = np.concatenate(
                    [
                        kpts[:, :2],
                        (np.hypot(kpts[:, 2], kpts[:, 4])[:, None] + np.hypot(kpts[:, 3], kpts[:, 5])[:, None]) * 0.5,
                        np.arctan2(kpts[:, 2:3], kpts[:, 4:5]) * 180 / np.pi,
                    ],
                    axis=1,
                )
            images_dict[image_id] = ImageFeature(image_name=image_name, keypoints=kpts)
        db.close()
        ImageFeature.set_order(images_dict)
        return images_dict

    @staticmethod
    def set_order(images: Dict[int, "ImageFeature"]):
        name_ids = [(im.image_name, im_id) for im_id, im in images.items()]
        sorted_items = sorted(name_ids)
        for order, (_, im_id) in enumerate(sorted_items):
            images[im_id].order = order


    def order_distance(self, other: 'ImageFeature') -> int:
        return abs(self.order - other.order)
