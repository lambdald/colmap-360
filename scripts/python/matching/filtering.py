#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import cv2 as cv
import pydegensac
import argparse
import sqlite3
import tqdm
from time import time
import shutil
import tyro
from typing import Dict, List, Set

from dataclasses import dataclass
from pathlib import Path
import shutil
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms as T
import cv2
from pathlib import Path
import argparse
import sys
import dataclasses
from matplotlib import pyplot as plt

from rich.progress import track
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from read_write_model import read_model, write_model, Camera, Image, Point3D, rotmat2qvec
from database import COLMAPDatabase, pair_id_to_image_ids, image_ids_to_pair_id
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import time
from multiple_view_matches import MultipleViewMatches, compare_view_matches
from tracks import FeaturePointTrack, FeaturePointsTracks



two_view_geo = []
two_size = 0
database_path_g = ""


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def array_to_blob(array):
    return array.tobytes()


def add_two_view_geometry(
    database_path: str, matches_result, qvec=np.array([1.0, 0.0, 0.0, 0.0]), tvec=np.zeros(3), config=2
):
    print(database_path)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM two_view_geometries")
    # cursor.execute("CREATE TABLE two_view_geometries   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,    rows     INTEGER               NOT NULL,    cols     INTEGER               NOT NULL,    data     BLOB,    config   INTEGER               NOT NULL,    F        BLOB,    E        BLOB,    H        BLOB,    qvec     BLOB,    tvec     BLOB)")
    for pair_id in tqdm.tqdm(matches_result.keys()):
        E = np.eye(3)
        F = matches_result[pair_id][1]
        H = matches_result[pair_id][2]

        matches = np.asarray(matches_result[pair_id][0], np.uint32)
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        cursor.execute(
            "INSERT OR REPLACE INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )
    connection.commit()
    cursor.close()
    connection.close()


def verify_pydegensac(kps1, kps2, tentatives, th=4.0, n_iter=2000):
    src_pts = kps1[tentatives[:, 0]]
    dst_pts = kps2[tentatives[:, 1]]
    H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    return H, mask


def verify_pydegensac_fundam(kps1, kps2, tentatives, th=1.0, n_iter=10000):
    src_pts = kps1[tentatives[:, 0]]
    dst_pts = kps2[tentatives[:, 1]]
    F, mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.999, n_iter, enable_degeneracy_check=True)
    return F, mask


def draw_matches(kps1, kps2, matches, image_path_prefix, image_name1, image_name2, output_path, output_name):
    if len(matches) < 15:
        return
    image1 = cv.imread(os.path.join(image_path_prefix, image_name1))
    image2 = cv.imread(os.path.join(image_path_prefix, image_name2))
    cv_keypoint1 = [cv.KeyPoint(pt[0], pt[1], 1) for pt in kps1]
    cv_keypoint2 = [cv.KeyPoint(pt[0], pt[1], 1) for pt in kps2]
    cv_matches = [cv.DMatch(match[0], match[1], 0) for match in matches]
    output_image = cv.drawMatches(image1, cv_keypoint1, image2, cv_keypoint2, cv_matches, None)
    cv.imwrite(os.path.join(output_path, output_name), output_image)







def filter_matches_pair(kps1, kps2, tentatives):

    kps1 = kps1[:, :2]
    kps2 = kps2[:, :2]
    src_pts = kps1[tentatives[:, 0]]
    dst_pts = kps2[tentatives[:, 1]]


    th=0.5
    n_iter=1000
    F, mask_F = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.999, n_iter, enable_degeneracy_check=True)
    inlier_F = np.sum(mask_F)

    return F, np.eye(3), tentatives[mask_F]

    th=1.0
    n_iter=1000
    # H, mask_H = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    H, mask_H = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, th, 0.99, n_iter)
    inlier_H = np.sum(mask_H)



    H_F_threshold = 0.8

    H_F_inlier_ratio = inlier_H / inlier_F


    geometry_mode = 'F'
    if H_F_inlier_ratio > H_F_threshold:
        geometry_mode = 'H'





    # mask = np.logical_and(mask_H, mask_F)
    mask = mask_F
    final_matches = tentatives[mask]
    return F, H, final_matches


def filter_matches(
    images: dict, keypoints: dict, matches: dict, image_path_prefix: str, output_path: str, database_path: str
):
    t = time()
    global two_view_geo
    global two_size
    global database_path_g
    pair_list = matches.keys()
    filter_results = {}
    print("start filtering")
    for pair_id in tqdm.tqdm(pair_list):
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        kps1, s1, o1 = keypoints[image_id1]
        kps2, s2, o2 = keypoints[image_id2]
        data = matches[pair_id]
        if data == None or data[2] == None:
            continue
        tentatives = blob_to_array(data[2], np.int32, (data[0], data[1]))
        matrix_F, matrix_H, final_macthes = filter_matches_pair(kps1=kps1, kps2=kps2, tentatives=tentatives)
        if final_macthes == None or len(final_macthes) < 15:
            continue
        filter_results[pair_id] = [final_macthes, matrix_F, matrix_H]

    add_two_view_geometry(database_path=database_path, matches_result=filter_results)

    print(f"Elapsed time: {time()-t}")


def draw_two_view(images: dict, keypoints: dict, two_view: dict, image_path_prefix: str, output_path: str):
    pair_list = two_view.keys()
    for pair_id in pair_list:
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        kps1, s1, o1 = keypoints[image_id1]
        kps2, s2, o2 = keypoints[image_id2]
        data = two_view[pair_id]
        if data == None or data[2] == None:
            continue
        tentatives = blob_to_array(data[2], np.int32, (data[0], data[1]))
        draw_matches(
            kps1=kps1,
            kps2=kps2,
            matches=tentatives,
            image_path_prefix=image_path_prefix,
            image_name1=images[image_id1],
            image_name2=images[image_id2],
            output_path=output_path,
            output_name=(str(int(image_id1 * 100 + image_id2)) + ".jpg"),
        )



@dataclass
class MatchFilterConfig:
    data_dir: Path
    input_database_name: Path
    output_database_name: Path
    image_reldir: Path = Path("images")
    debug_reldir: Path = Path("debug_viz/filter_matches")


    num_adjacent_frames: int = 50
    loop_importance_ratio: float = 0.2
    num_longest_track: int = 500


    filtering_by_loop_importance: bool = False

    filtering_by_track: bool = False
    filtering_long_track: bool = False



class MatchFilter:
    def __init__(self, config: MatchFilterConfig) -> None:
        self.config = config

    def get_abspath(self, path) -> Path:
        return self.config.data_dir / path

    def copy_db(self, src_db, dst_db):
        shutil.copy(src_db, dst_db)

    def load_data_from_colmap_database(self, db_path: Path) -> MultipleViewMatches:
        return MultipleViewMatches.from_colmap(db_path)



    def filter_two_view_matches(self, kpts0, kpts1, matches):
        F, H, inlier_matches = filter_matches_pair(kpts0, kpts1, matches)
        return F, H, inlier_matches



    def filter_loop_matches(self, two_view_geo: MultipleViewMatches) -> MultipleViewMatches:
        
        max_num_inliers = {}

        for (id1, id2) in two_view_geo.matches:
            match = two_view_geo.matches[(id1, id2)]
            inlier_matches = match.inlier_matches

            max_num_inliers[id1] = max(max_num_inliers.get(id1, 0), len(inlier_matches))
            max_num_inliers[id2] = max(max_num_inliers.get(id2, 0), len(inlier_matches))


        new_matches = MultipleViewMatches(images=two_view_geo.images, matches={})

        for (id1, id2) in two_view_geo.matches:
            match = two_view_geo.matches[(id1, id2)]

            if match.type == 'adjacent':
                new_matches.matches[(id1, id2)] = match
                continue

            inlier_matches = match.inlier_matches

            ratio_1 = len(inlier_matches) / max_num_inliers[id1]
            ratio_2 = len(inlier_matches) / max_num_inliers[id2]

            importance_value = min(ratio_1, ratio_2)

            if importance_value > self.config.loop_importance_ratio:
                new_matches.matches[(id1, id2)] = match

        return new_matches


    def filter_matches(self, two_view_matches: MultipleViewMatches) -> MultipleViewMatches:
        new_matches = MultipleViewMatches(images=two_view_matches.images, matches={})

        num_worker = mp.cpu_count()
        print('num worker=', num_worker)
        # num_worker = None
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            futures = {}
            cnt = 0

            for id0, id1 in tqdm.tqdm(two_view_matches.matches, desc='filter matches', total=len(two_view_matches.matches)):

                kpts0 = two_view_matches.images[id0]['kpts']
                kpts1 = two_view_matches.images[id1]['kpts']
                matches = two_view_matches.matches[(id0, id1)]['matches']
                f = executor.submit(filter_matches_pair, kpts0, kpts1, matches)
                futures[f] = (id0, id1)
                cnt += 1

                if len(futures) >= 1000:
                    for f in as_completed(futures):
                        assert f.done()
                        id0, id1 = futures[f]
                        # print(id0, id1)
                        F, H, inlier_matches = f.result(timeout=10)
                        # print(id0, id1, f)
                        if inlier_matches is None or len(inlier_matches) < 30:
                            continue

                        colmap_inlier_matches = two_view_matches.matches[(id0, id1)]['inlier_matches']

                        # print(f'matches: {len(colmap_inlier_matches)} -> {len(inlier_matches)}')

                        new_matches.matches[(id0, id1)] = {
                            "matches": matches,
                            "inlier_matches": inlier_matches,
                            "F": F,
                            "H": H
                        }
                    print(f'{cnt} clean up thread pool')
                    futures = {}

            for f in as_completed(futures):
                assert f.done()
                id0, id1 = futures[f]
                print(id0, id1)
                F, H, inlier_matches = f.result(timeout=10)
                print(id0, id1, f)
                if inlier_matches is None or len(inlier_matches) < 30:
                    continue
                new_matches.matches[(id0, id1)] = {
                    "matches": matches,
                    "inlier_matches": inlier_matches,
                    "F": F,
                    "H": H
                }
            executor.shutdown()
        return new_matches

    def export_matches_to_database(self, database_path: str, matches: MultipleViewMatches, qvec=np.array([1.0, 0.0, 0.0, 0.0]), tvec=np.zeros(3), config=3):
        print(f'export match result to {database_path}')
        db = COLMAPDatabase.connect(database_path)
        db.execute("DELETE FROM two_view_geometries")
        for im_id0, im_id1 in tqdm.tqdm(matches.matches.keys()):

            pair_id = image_ids_to_pair_id(im_id0, im_id1)
            two_view_geo = matches.matches[(im_id0, im_id1)]

            E = np.eye(3)
            F = two_view_geo.F
            H = np.eye(3)

            inlier_matches = two_view_geo.inlier_matches.astype(np.uint32)

            assert len(inlier_matches.shape) == 2
            assert inlier_matches.shape[1] == 2

            F = np.asarray(F, dtype=np.float64)
            E = np.asarray(E, dtype=np.float64)
            H = np.asarray(H, dtype=np.float64)
            qvec = np.asarray(qvec, dtype=np.float64)
            tvec = np.asarray(tvec, dtype=np.float64)
            db.execute(
                "INSERT OR REPLACE INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id,)
                + inlier_matches.shape
                + (
                    array_to_blob(inlier_matches),
                    config,
                    array_to_blob(F),
                    array_to_blob(E),
                    array_to_blob(H),
                    array_to_blob(qvec),
                    array_to_blob(tvec),
                ),
            )
        db.commit()
        db.close()


    def filter_by_track(self, multi_view_matches: MultipleViewMatches) -> MultipleViewMatches:
        print('filter by ambiguity tracks and very long track')
        debug_dir = self.get_abspath(self.config.debug_reldir)

        tracks = FeaturePointsTracks.construct_tracks(multi_view_matches)
        tracks.save_track(debug_dir/'tracks.png')
        tracks = tracks.filter_ambiguity_track()
        tracks.save_track(debug_dir/'tracks_filtered_ambiguity.png')


        if self.config.filtering_long_track:
            tracks = tracks.filter_long_track(self.config.num_longest_track)
            tracks.save_track(debug_dir/'tracks_filtered_ambiguity_long.png')

        query = tracks.build_query_table()
        new_multi_view_matches = query.filter_matches(multi_view_matches)
        new_multi_view_matches.save_match_graph(debug_dir/'match_graph_filtered_track.png')
        print(new_multi_view_matches)

        diff = compare_view_matches(multi_view_matches, new_multi_view_matches)
        diff = np.uint8(diff / diff.max() * 255)
        cv2.imwrite(str(debug_dir/'match_graph_filtered_track_diff.png'), diff)
        return new_multi_view_matches

    def filter_by_loop_importance(self, multi_view_matches: MultipleViewMatches) -> MultipleViewMatches:
        print('filter matches by loop matches importance')
        multi_view_matches.classify_matches(self.config.num_adjacent_frames)

        debug_dir = self.get_abspath(self.config.debug_reldir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        multi_view_matches.save_match_graph(debug_dir/'match_graph.png')
        multi_view_matches_filtered_loop = self.filter_loop_matches(multi_view_matches)
        multi_view_matches_filtered_loop.save_match_graph(debug_dir/'match_graph_filtered_loop.png')
        print('Filtering by mathes', multi_view_matches_filtered_loop)

        diff = compare_view_matches(multi_view_matches, multi_view_matches_filtered_loop)
        diff = np.uint8(diff / diff.max() * 255)
        cv2.imwrite(str(debug_dir/'match_graph_filtered_loop_diff.png'), diff)

        return multi_view_matches_filtered_loop

    def run(self):
        self.copy_db(
            self.get_abspath(self.config.input_database_name), self.get_abspath(self.config.output_database_name)
        )
        multi_view_matches = self.load_data_from_colmap_database(self.get_abspath(self.config.input_database_name))
        print('init matches', multi_view_matches)

        if self.config.filtering_by_loop_importance:
            multi_view_matches = self.filter_by_loop_importance(multi_view_matches)

        if self.config.filtering_by_track:
            multi_view_matches = self.filter_by_track(multi_view_matches)

        multi_view_matches = multi_view_matches.estimate_two_view_geometry()
        self.export_matches_to_database(self.get_abspath(self.config.output_database_name), multi_view_matches)
        print('done')

if __name__ == "__main__":
    mp.set_start_method('spawn')
    config = tyro.cli(MatchFilterConfig)
    filter = MatchFilter(config)
    filter.run()
