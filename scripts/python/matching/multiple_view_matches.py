from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from database import COLMAPDatabase, pair_id_to_image_ids
from rich.progress import track
from image.image import ImageFeature
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import pydegensac


@dataclass
class TwoViewMatches:
    view_id0: int
    view_id1: int
    matches: np.ndarray
    inlier_matches: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None

    type: Literal["adjacent", "loop", "unknown"] = "unknown"




def find_fundamental_matrix(kps1, kps2, tentatives, num_iter=1000, threshold=0.5):

    kps1 = kps1[:, :2]
    kps2 = kps2[:, :2]
    src_pts = kps1[tentatives[:, 0]]
    dst_pts = kps2[tentatives[:, 1]]
    F, mask_F = pydegensac.findFundamentalMatrix(src_pts, dst_pts, threshold, 0.999, num_iter, enable_degeneracy_check=True)
    return F,  tentatives[mask_F]


@dataclass
class MultipleViewMatches:
    images: Dict[int, ImageFeature]
    matches: Dict[Tuple[int, int], TwoViewMatches]

    @staticmethod
    def from_colmap(database_path: Path) -> "MultipleViewMatches":

        images = ImageFeature.from_colmap(database_path)

        db = COLMAPDatabase.connect(database_path)
        rows = db.execute("SELECT * FROM matches")

        two_view_matches = {}
        for pair_id, r, c, data in track(rows, description="load matches"):
            if data is None:
                continue
            id1, id2 = pair_id_to_image_ids(pair_id)
            id1 = int(id1)
            id2 = int(id2)

            matches = np.frombuffer(data, np.uint32).reshape(r, c)
            two_views = next(db.execute(f"SELECT * FROM two_view_geometries WHERE pair_id={pair_id}"))

            _, n, d, inlier_matches, config, F, E, H, qvec, tvec = two_views
            if inlier_matches is not None:
                inlier_matches = np.frombuffer(inlier_matches, np.uint32).reshape(n, d)
                F = np.frombuffer(F, np.float64).reshape(3, 3)
                E = np.frombuffer(E, np.float64).reshape(3, 3)
                H = np.frombuffer(H, np.float64).reshape(3, 3)
                two_view_matches[(id1, id2)] = TwoViewMatches(
                    view_id0=id1, view_id1=id2, matches=matches, inlier_matches=inlier_matches, F=F, E=E, H=H
                )

            # if len(two_view_matches) > 10000:
            #     break

        db.close()
        return MultipleViewMatches(images=images, matches=two_view_matches)

    def draw_match_graph(self):
        max_id = max(self.images.keys()) + 1
        view_graph = np.zeros((max_id, max_id), dtype=np.uint32)
        # for image_id in self.images:
        #     view_graph[image_id, image_id] = self.images[image_id].keypoints.shape[0]

        for id0, id1 in self.matches:
            if self.matches[(id0, id1)].inlier_matches is not None:
                num = self.matches[(id0, id1)].inlier_matches.shape[0]
            else:
                num = self.matches[(id0, id1)].matches.shape[0]
            view_graph[id0, id1] = num
            view_graph[id1, id0] = num
        return view_graph

    def save_match_graph(self, image_path):
        view_graph = self.draw_match_graph()
        view_graph = (view_graph / view_graph.max() * 255).astype(np.uint8)
        cv2.imwrite(str(image_path), view_graph)


    def num_images(self):
        return len(self.images)

    def num_matches(self):
        return len(self.matches)

    def __str__(self):
        return f"MultipleViewMatches with {self.num_images()} images and {self.num_matches()} matches"

    def classify_matches(self, num_adjacent_frames: int):
        '''adjacent or loop'''
        for id1, id2 in self.matches:
            im1 = self.images[id1]
            im2 = self.images[id2]

            if im1.order_distance(im2) < num_adjacent_frames:
                self.matches[(id1, id2)].type = "adjacent"
            else:
                self.matches[(id1, id2)].type = "loop"

    def estimate_two_view_geometry(self):
        new_matches = MultipleViewMatches(images=self.images, matches={})

        num_worker = mp.cpu_count()
        print('num worker=', num_worker)
        # num_worker = None
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            futures = {}
            cnt = 0
            def clear_futures(futures:Dict):
                for f in as_completed(futures):
                    assert f.done()
                    id0, id1 = futures[f]
                    F, inlier_matches = f.result(timeout=10)
                    if inlier_matches is None or len(inlier_matches) < 10:
                        continue
                    new_matches.matches[(id0, id1)] = TwoViewMatches(
                        view_id0=two_view_match.view_id0,
                        view_id1=two_view_match.view_id1,
                        matches=two_view_match.matches,
                        inlier_matches=inlier_matches,
                        F=F,
                        type=two_view_match.type
                    )
                futures.clear()

            for id0, id1 in tqdm.tqdm(self.matches, desc='extimate two view geometry', total=len(self.matches)):

                kpts0 = self.images[id0].keypoints
                kpts1 = self.images[id1].keypoints

                two_view_match = self.matches[(id0, id1)]
                f = executor.submit(find_fundamental_matrix, kpts0, kpts1, two_view_match.matches)
                futures[f] = (id0, id1)
                cnt += 1
                if len(futures) >= 1000:
                    clear_futures(futures)
                    
            clear_futures(futures)
            executor.shutdown()
        return new_matches


def compare_view_matches(match1: MultipleViewMatches, match2: MultipleViewMatches):
    view_graph1 = match1.draw_match_graph()
    view_graph2 = match2.draw_match_graph()
    diff = np.abs(view_graph1 - view_graph2).astype(np.uint32)
    return diff
