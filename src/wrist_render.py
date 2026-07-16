"""MVP item 2: rotate the RM65 wrist through a few configurations, publish
each to Meshcat, and render an RGB frame from a fixed simulated camera at
each step. Frames are saved to src/data/pretrain_frames/ as a tiny image
dataset stub for later perception pretraining.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image
from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRenderCamera,
    DepthRange,
    MakeRenderEngineVtk,
    Meshcat,
    RenderCameraCore,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from scene import ARM_URDF, add_meshcat_visualizer
from airo_drake import finish_build
from pydrake.planning import RobotDiagramBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = Path(__file__).resolve().parent / "data" / "pretrain_frames"

# Wrist joint (joint_6, the 6th/last RM65 joint) sweep, in radians.
WRIST_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 5)

BASE_ARM_CONFIGURATION = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


def _add_fixed_camera(robot_diagram_builder: RobotDiagramBuilder, renderer_name: str = "vtk_renderer"):
    scene_graph = robot_diagram_builder.scene_graph()
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

    intrinsics = CameraInfo(width=640, height=480, fov_y=np.pi / 4)
    clipping = ClippingRange(0.01, 10.0)
    color_camera = ColorRenderCamera(RenderCameraCore(renderer_name, intrinsics, clipping, RigidTransform()))
    depth_camera = DepthRenderCamera(
        RenderCameraCore(renderer_name, intrinsics, clipping, RigidTransform()), DepthRange(0.01, 10.0)
    )

    plant = robot_diagram_builder.plant()
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    # Looking at the arm from ~1m away, slightly above base height.
    X_W_Cam = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2), [0.9, 0.0, 0.6])

    builder = robot_diagram_builder.builder()
    sensor = builder.AddSystem(RgbdSensor(world_id, X_W_Cam, color_camera, depth_camera))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    return sensor


def build_scene_with_camera(meshcat: Meshcat):
    """RM65-only scene (no hand) plus a fixed RGB-D camera.

    The wrist-rotation render demo only needs the arm, so the hand is left
    out here (it's still used in grab_demo.py, which only visualizes through
    Meshcat and never renders it).
    """
    robot_diagram_builder = RobotDiagramBuilder()
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)

    meshcat.Delete()
    meshcat.DeleteAddedControls()
    add_meshcat_visualizer(robot_diagram_builder, meshcat)
    sensor = _add_fixed_camera(robot_diagram_builder)

    arm_index = parser.AddModels(str(ARM_URDF))[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", arm_index))

    robot_diagram, context = finish_build(robot_diagram_builder, meshcat)
    return robot_diagram, context, plant, arm_index, sensor


def run_wrist_render(meshcat: Meshcat, publish_delay: float = 0.3, save_frames: bool = True):
    """Rotates the wrist through WRIST_ANGLES, publishing to Meshcat and rendering+saving a frame each step.

    Returns the list of rendered RGB frames (H, W, 4 uint8 arrays).
    """
    robot_diagram, context, plant, arm_index, sensor = build_scene_with_camera(meshcat)
    plant_context = plant.GetMyContextFromRoot(context)
    sensor_context = sensor.GetMyContextFromRoot(context)

    if save_frames:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, wrist_angle in enumerate(WRIST_ANGLES):
        q = BASE_ARM_CONFIGURATION.copy()
        q[-1] = wrist_angle
        plant.SetPositions(plant_context, arm_index, q)
        robot_diagram.ForcedPublish(context)
        time.sleep(publish_delay)

        image = sensor.color_image_output_port().Eval(sensor_context)
        frame = np.array(image.data, copy=True)
        frames.append(frame)

        if save_frames:
            out_path = FRAMES_DIR / f"frame_{i:02d}_wrist_{np.rad2deg(wrist_angle):+.0f}deg.png"
            Image.fromarray(frame[..., :3]).save(out_path)

    return frames


if __name__ == "__main__":
    meshcat = Meshcat()
    print(f"Meshcat running at {meshcat.web_url()}")
    frames = run_wrist_render(meshcat)
    print(f"Saved {len(frames)} frames to {FRAMES_DIR}")
    input("Press Enter to exit...")
