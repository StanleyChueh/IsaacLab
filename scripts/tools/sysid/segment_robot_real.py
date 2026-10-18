# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Segment the robot in real camera frames, for silhouette-based camera calibration.

Grounding DINO boxes the arms from a text prompt, SAM 2.1 turns each box into a mask, and the
manipulated object is segmented the same way and cut out (a held can sits inside the gripper's box and
SAM happily includes it). A morphological opening then drops the cables, which the sim robot does not
have. Writes <frames dir>/robot_masks.npz (``masks`` (N,H,W) bool) and mask_XX.png previews.

Both models are ungated on Hugging Face (SAM 3 would take the text prompt directly but is gated). Needs
transformers >= 5 -- lerobot_openarm's venv has it:
  ~/Stanley_ws/lerobot_openarm/.venv/bin/python3 scripts/tools/sysid/segment_robot_real.py \\
      logs/real2sim/body_cam_frames/frames.npz
"""

import argparse
import os

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection, Sam2Model, Sam2Processor


class Segmenter:
    def __init__(self, device: str):
        self.device = device
        self.gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.gm = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        self.sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, text: str, threshold: float, max_box: float) -> np.ndarray:
        """Union of the masks of every `text` detection whose box covers at most `max_box` of the image."""
        inputs = self.gp(images=image, text=text, return_tensors="pt").to(self.device)
        det = self.gp.post_process_grounded_object_detection(
            self.gm(**inputs), inputs.input_ids, threshold=threshold, text_threshold=threshold,
            target_sizes=[image.shape[:2]])[0]
        mask = np.zeros(image.shape[:2], bool)
        h, w = image.shape[:2]
        boxes = [b for b in det["boxes"].cpu().numpy().tolist() if (b[2] - b[0]) * (b[3] - b[1]) <= max_box * h * w]
        if boxes:
            inputs = self.sp(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            out = self.sm(**inputs, multimask_output=False)
            mask = self.sp.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][:, 0].numpy().any(0)
        return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="frames.npz from extract_real_frames.py")
    ap.add_argument("--robot_prompt", default="robot arm. robot gripper.")
    ap.add_argument("--object_prompt", default="pringles can.")
    ap.add_argument("--threshold", type=float, default=0.25)
    # A box spanning the whole table makes SAM segment the mat (seen at 0.7-0.98 of the image); one arm's
    # box is <= ~0.45, and the can's <= ~0.1 (a 0.7 "can" box once covered both grippers instead).
    ap.add_argument("--max_robot_box", type=float, default=0.6, help="Ignore robot boxes larger than this image fraction.")
    ap.add_argument("--max_object_box", type=float, default=0.15, help="Ignore object boxes larger than this image fraction.")
    ap.add_argument("--cable_px", type=int, default=9, help="Opening kernel; removes anything thinner (cables).")
    args = ap.parse_args()

    images = np.load(args.frames)["images"]
    seg = Segmenter("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(os.path.abspath(args.frames))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.cable_px, args.cable_px))
    masks = []
    for k, image in enumerate(images):
        robot = seg(image, args.robot_prompt, args.threshold, args.max_robot_box)
        obj = cv2.dilate(seg(image, args.object_prompt, args.threshold, args.max_object_box).astype(np.uint8), kernel) > 0
        mask = cv2.morphologyEx((robot & ~obj).astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        masks.append(mask)
        vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        vis[mask] = (0.5 * vis[mask] + (0, 0, 127)).astype(np.uint8)
        vis[obj] = (0.5 * vis[obj] + (127, 0, 0)).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"mask_{k:02d}.png"), vis)
        print(f"frame {k}: robot {mask.mean() * 100:.1f}% of the image, object {obj.mean() * 100:.1f}%")
    np.savez_compressed(os.path.join(out_dir, "robot_masks.npz"), masks=np.stack(masks))
    print(f"-> {out_dir}/robot_masks.npz")


if __name__ == "__main__":
    main()
