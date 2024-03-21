import numpy as np
from multiple_view_matches import MultipleViewMatches, TwoViewMatches
import tqdm
from typing import Set, Tuple, Dict, List, Optional
from dataclasses import dataclass
import copy

from matplotlib import pyplot as plt
from collections import Counter

from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from multiprocessing import cpu_count

class UnionFind2D:
    def __init__(self):
        self.data = {}

    def find(self, key):
        while key in self.data:
            
            next_key = self.data[key]
            if next_key == key:
                break
            
            if next_key in self.data:
                self.data[key] = self.data[next_key]
            key = next_key


        if key not in self.data:
            self.data[key] = key
        return key

    def union(self, id0: int, id1: int, matches: np.ndarray):

        for m in matches:

            p0 = (id0, m[0])
            p1 = (id1, m[1])

            r0 = self.find(p0)
            r1 = self.find(p1)

            if r0 == r1:
                continue
            self.data[r0] = r1

    def path(self, key):
        path = [key]
        while key in self.data:
            key = self.data[key]
            path.append(key)
        return path

    def gather(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        groups = {}

        for key in self.data:
            root = self.find(key)
            if root not in groups:
                groups[root] = set()
            groups[root].add(key)
        return groups


class FeaturePointTrack:
    '''
    Tracking of same scene point.
    '''
    view_track: Dict[int, Set]
    track: Set[Tuple[int, int]]
    def __init__(self, track: Set[Tuple[int, int]]):
        self.track = track
        self.build_view_track(filter_ambiguity=False)

    def filter_ambiguity_views(self) -> Set[int]:
        # filter anbiguity loop track
        ambiguity_views = set()
        new_view_track: Dict[int, Set[int]] = {}
        for view_id in self.view_track:
            if len(self.view_track[view_id]) == 1:
                new_view_track[view_id] = self.view_track[view_id]
            else:
                ambiguity_views.add(view_id)
        self.view_track = new_view_track
        self.track = set()
        for view_id in self.view_track:
            point_id = self.view_track[view_id].pop()
            self.track.add((view_id, point_id))
        return ambiguity_views

    def build_view_track(self, filter_ambiguity=False):
        self.view_track: Dict[int, Set] = {}
        for view_id, feat_id in self.track:
            if view_id not in self.view_track:
                self.view_track[view_id] = set()
            self.view_track[view_id].add(feat_id)

    def __len__(self):
        return len(self.view_track)

    def __str__(self):
        return f"FeaturePointTrack with length {len(self)}"


@dataclass
class FeatureMatchesQuery:
    query: Dict[int, Dict[int, FeaturePointTrack]]

    def filter_match(self, view_id0: int, view_id1: int, match: TwoViewMatches) -> Optional[TwoViewMatches]:
        new_matches = []
        for m in match.matches:
            point_id0 = m[0]
            point_id1 = m[1]
            
            kpts0 = self.query.get(view_id0, None)
            kpts1 = self.query.get(view_id1, None)

            if kpts0 is None or kpts1 is None:
                continue

            track0 = kpts0.get(point_id0, None)
            track1 = kpts1.get(point_id1, None)

            if track0 is not None and track1 is not None:
                if id(track0) == id(track1):
                    new_matches.append((point_id0, point_id1))

 

        if len(new_matches) < 10:
            return None
        
        new_matches = np.array(new_matches)
        return TwoViewMatches(
            view_id0=view_id0, view_id1=view_id1, matches=new_matches, inlier_matches=new_matches, F=match.F, type=match.type
        )



    def filter_matches(self, matches: MultipleViewMatches) -> MultipleViewMatches:
        new_matches = {}
        for (view_id0, view_id1), match in tqdm.tqdm(matches.matches.items(), desc='filter matches'):
            new_match = self.filter_match(view_id0, view_id1, match)
            if new_match is not None:
                new_matches[(view_id0, view_id1)] = new_match
        # def clear_futures(futures:Dict):
        #     for f in as_completed(futures):
        #         assert f.done()
        #         (view_id0, view_id1) = futures[f]
        #         new_match = f.result()
        #         if new_match is not None:
        #             new_matches[(view_id0, view_id1)] = new_match
        #     futures.clear()

        # with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
        #     futures = {}
        #     for (view_id0, view_id1), match in tqdm.tqdm(matches.matches.items(), desc='filter matches'):
        #         f = executor.submit(self.filter_match, view_id0, view_id1, match)
        #         futures[f] = (view_id0, view_id1)
        #         if len(futures) > 1000:
        #             clear_futures(futures)

        #     clear_futures(futures)

        return MultipleViewMatches(
            images=matches.images, matches=new_matches
        )


@dataclass
class FeaturePointsTracks:
    '''Collection of scene point tracking.'''
    tracks: List[FeaturePointTrack]

    def get_max_num_features(self):
        return max(len(im["kpts"]) for im in self.matches.images.values())

    @staticmethod
    def construct_tracks(multiple_view_matches: MultipleViewMatches) -> "FeaturePointsTracks":
        union = UnionFind2D()
        for (id0, id1), kpt_match in tqdm.tqdm(multiple_view_matches.matches.items(), desc="union two view matches"):
            inlier_matches = kpt_match.inlier_matches
            union.union(id0, id1, inlier_matches)
        tracks_set = union.gather()
        tracks = FeaturePointsTracks(tracks=[])

        for _, track_set in tqdm.tqdm(tracks_set.items(), 'build track'):
            track = FeaturePointTrack(track_set)
            tracks.tracks.append(track)

        return tracks

    def filter_long_track(self, max_len: int) -> "FeaturePointsTracks":
        new_tracks = []
        for track in tqdm.tqdm(self.tracks, desc='Filter long tracks'):
            if len(track) <= max_len:
                new_tracks.append(track)
        return FeaturePointsTracks(tracks=new_tracks)

    def filter_ambiguity_track(self) -> "FeaturePointsTracks":
        new_tracks = []
        ambiguity_counts = Counter()
        for track in tqdm.tqdm(self.tracks, desc='Filter ambiguity tracks'):
            new_track = copy.copy(track)
            ambiguity_views = new_track.filter_ambiguity_views()
            ambiguity_counts.update(ambiguity_views)
            if len(new_track) > 1:
                new_tracks.append(new_track)
        print(ambiguity_counts.most_common(10))
        return FeaturePointsTracks(new_tracks)

    def save_track(self, filepath, max_track=-1):
        max_num_track = max(len(t) for t in self.tracks) + 1
        print("max_num_track=", max_num_track)
        sta = np.zeros((max_num_track,), dtype=int)
        for track in self.tracks:
            track_len = len(track)
            sta[track_len] += 1

        if max_track <= 0:
            max_track = max_num_track

        plt.bar(np.arange(max_num_track)[:max_track], sta[:max_track])
        plt.title(f"Feature Track[max_track={max_num_track} mean={sta.mean()}]")
        plt.savefig(str(filepath))
        plt.close()

    def build_query_table(self)->FeatureMatchesQuery:
        query = {}
        for track in tqdm.tqdm(self.tracks, desc='buile query table'):
            for view_id, point_id in track.track:
                if view_id not in query:
                    query[view_id] = {}
    
                query[view_id][point_id] = track

        return FeatureMatchesQuery(query=query)
