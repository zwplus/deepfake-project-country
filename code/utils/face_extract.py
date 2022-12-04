import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

from .augmentations import letterbox


class FaceExtractor:
    """Wrapper for face extraction workflow."""

    def __init__(self, video_read_fn=None, facedet=None, batch_size=32):
        """Creates a new FaceExtractor.

        Arguments:
            video_read_fn: a function that takes in a path to a video file
                and returns a tuple consisting of a NumPy array with shape
                (num_frames, H, W, 3) and a list of frame indices, or None
                in case of an error
            facedet: the face detector object
        """
        self.video_read_fn = video_read_fn
        self.facedet = facedet
        self.batch_size = batch_size

    def process_image(self, path: str = None, img: Image.Image or np.ndarray = None) -> list:
        """
        Process a single image
        :param path: Path to the image
        :param img: image
        :return:
        """

        if img is not None and path is not None:
            raise ValueError('Only one argument between path and img can be specified')
        if img is None and path is None:
            raise ValueError('At least one argument between path and img must be specified')

        if img is None:
            img=cv2.imread(str(path))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = np.asarray(Image.open(str(path)).convert('RGB'))
        else:
            img = np.asarray(img)
        return self._process_images([img])[0]

    def _process_images(self, frames, frames_idx=None) -> list:
        if not frames_idx:
            frames_idx = list(range(len(frames)))
        assert len(frames) == len(frames_idx)
        target_size = self.facedet.input_size
        inputs = []
        inputs_ratio = []
        inputs_size = []
        for frame in frames:
            _frame, _ratio, _size = letterbox(frame, target_size)
            inputs.append(_frame)
            inputs_ratio.append(_ratio)
            inputs_size.append(_size)
        detections = []
        for i in range(0, len(inputs), self.batch_size):
            _inputs = np.stack(inputs[i: i + self.batch_size])
            _detections = self.facedet.predict_on_batch(_inputs)
            detections.extend(_detections)
        results = []
        for detection, ratio, size, frame, idx in zip(detections, inputs_ratio, inputs_size, frames, frames_idx):
            detection = detection.cpu()
            dw, dh = size
            _p = np.array([dh, dw, dh, dw])
            rw, rh = ratio
            _r = np.array([rh, rw, rh, rw])
            h, w = target_size
            _s = np.array([h, w, h, w])
            detection[:, : 4] = (detection[:, : 4] * _s - _p) / _r
            fh, fw = frame.shape[: 2]
            frameref_detection = self._add_margin_to_detections(detection, (fw, fh), 0.2)
            faces = self._crop_faces(frame, frameref_detection)
            scores = list(detection[:, -1].cpu().numpy())
            frame_dict = {
                "frame_idx": idx,
                "frame_w": fw,
                "frame_h": fh,
                "frame": frame,
                "faces": faces,
                "detections": frameref_detection.cpu().numpy(),
                "scores": scores,
            }
            # Sort faces by descending confidence
            frame_dict = self._soft_faces_by_descending_score(frame_dict)
            results.append(frame_dict)
        return results

    def _soft_faces_by_descending_score(self, frame_dict: dict) -> dict:
        if len(frame_dict['scores']) > 1:
            sort_idxs = np.argsort(frame_dict['scores'])[::-1]
            for key in ['faces', 'detections', 'scores']:
                frame_dict[key] = [frame_dict[key][i] for i in sort_idxs]
        return frame_dict

    def process_video(self, video_path):
        # Read the full-size frames from this video.
        video_read = self.video_read_fn(video_path)
        if video_read is None:
            return None
        # Keep track of the original frames (need them later).
        frames, frames_idx = video_read
        return self._process_images(frames, frames_idx)

    def _add_margin_to_detections(self, detections: torch.Tensor, frame_size: Tuple[int, int],
                                  margin: float = 0.2) -> torch.Tensor:
        """Expands the face bounding box.

        NOTE: The face detections often do not include the forehead, which
        is why we use twice the margin for ymin.

        Arguments:
            detections: a PyTorch tensor of shape (num_detections, 17)
            frame_size: maximum (width, height)
            margin: a percentage of the bounding box's height

        Returns a PyTorch tensor of shape (num_detections, 17).
        """
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset * 2, min=0)  # ymin
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)  # xmin
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
        return detections[:, : 4]

    def _crop_faces(self, frame: np.ndarray, detections: torch.Tensor) -> List[np.ndarray]:
        """Copies the face region(s) from the given frame into a set
        of new NumPy arrays.

        Arguments:
            frame: a NumPy array of shape (H, W, 3)
            detections: a PyTorch tensor of shape (num_detections, 17)

        Returns a list of NumPy arrays, one for each face crop. If there
        are no faces detected for this frame, returns an empty list.
        """
        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int64)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces

    def _crop_kpts(self, frame: np.ndarray, detections: torch.Tensor, face_fraction: float):
        """Copies the parts region(s) from the given frame into a set
        of new NumPy arrays.

        Arguments:
            frame: a NumPy array of shape (H, W, 3)
            detections: a PyTorch tensor of shape (num_detections, 17)
            face_fraction: float between 0 and 1 indicating how big are the parts to be extracted w.r.t the whole face

        Returns a list of NumPy arrays, one for each face crop. If there
        are no faces detected for this frame, returns an empty list.
        """
        faces = []
        for i in range(len(detections)):
            kpts = []
            size = int(face_fraction * min(detections[i, 2] - detections[i, 0], detections[i, 3] - detections[i, 1]))
            kpts_coords = detections[i, 4:16].cpu().numpy().astype(np.int)
            for kpidx in range(6):
                kpx, kpy = kpts_coords[kpidx * 2:kpidx * 2 + 2]
                kpt = frame[kpy - size // 2:kpy - size // 2 + size, kpx - size // 2:kpx - size // 2 + size, ]
                kpts.append(kpt)
            faces.append(kpts)
        return faces

    def remove_large_crops(self, crops, pct=0.1):
        """Removes faces from the results if they take up more than X%
        of the video. Such a face is likely a false positive.

        This is an optional postprocessing step. Modifies the original
        data structure.

        Arguments:
            crops: a list of dictionaries with face crop data
            pct: maximum portion of the frame a crop may take up
        """
        for i in range(len(crops)):
            frame_data = crops[i]
            video_area = frame_data["frame_w"] * frame_data["frame_h"]
            faces = frame_data["faces"]
            scores = frame_data["scores"]
            new_faces = []
            new_scores = []
            for j in range(len(faces)):
                face = faces[j]
                face_H, face_W, _ = face.shape
                face_area = face_H * face_W
                if face_area / video_area < 0.1:
                    new_faces.append(face)
                    new_scores.append(scores[j])
            frame_data["faces"] = new_faces
            frame_data["scores"] = new_scores

    def keep_only_best_face(self, crops):
        """For each frame, only keeps the face with the highest confidence.

        This gets rid of false positives, but obviously is problematic for
        videos with two people!

        This is an optional postprocessing step. Modifies the original
        data structure.
        """
        for i in range(len(crops)):
            frame_data = crops[i]
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]

    # TODO: def filter_likely_false_positives(self, crops):
    #   if only some frames have more than 1 face, it's likely a false positive
    #   if most frames have more than 1 face, it's probably two people
    #   so find the % of frames with > 1 face; if > 0.X, keep the two best faces

    # TODO: def filter_by_score(self, crops, min_score) to remove any
    # crops with a confidence score lower than min_score

    # TODO: def sort_by_histogram(self, crops) for videos with 2 people.

