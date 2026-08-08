import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType
from src.primary.color import COLOR_SPECS


DEFAULT_OBJECT_TYPE = ObjectType.PAPER_PLANE_SHAPES
PLANE_MARGIN_M = 0.015
OBJECT_AXIS_LENGTH_M = 0.08
PLANE_AXIS_LENGTH_M = 0.025


def transformPlanePoints(points_xy: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points_plane = np.column_stack((points_xy, np.zeros(len(points_xy))))
    return (rotation@points_plane.T).T + translation


def setAxesEqual(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs)/2.0
    radius = max(np.max(maxs - mins)/2.0, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualizeObjectVisionModel(object_type: ObjectType) -> None:
    spec = OBJECT_VISION_SPECS[object_type]

    if not spec.rigid_planes:
        raise ValueError(f"{object_type.name} has no rigid-plane model to visualize")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_points = [np.zeros(3)]

    # Object/reference-frame axes: +x forward, +y down, +z right.
    object_axis_vectors = np.eye(3)*OBJECT_AXIS_LENGTH_M
    object_axis_labels = ("+x forward", "+y down", "+z right")
    object_axis_colors = ("r", "g", "b")

    for vector, label, color in zip(object_axis_vectors, object_axis_labels, object_axis_colors):
        ax.quiver(0, 0, 0, *vector, color=color, linewidth=2, arrow_length_ratio=0.12)
        ax.text(*(vector*1.08), label, color=color)

    ax.scatter([0], [0], [0], color="k", s=35)
    ax.text(0, 0, 0, " object origin")

    for plane_index, rigid_plane in enumerate(spec.rigid_planes):
        rotation = rigid_plane.rotation_object_from_plane
        translation = rigid_plane.translation_object_from_plane_m
        all_points.append(translation)

        # Show the rigid plane's local x/y axes and normal after rotation into the object frame.
        local_axes_object = rotation@np.eye(3)
        local_labels = (f"P{plane_index} +x", f"P{plane_index} +y", f"P{plane_index} normal")
        local_colors = ("tab:red", "tab:green", "tab:blue")

        for axis_index in range(3):
            vector = local_axes_object[:, axis_index]*PLANE_AXIS_LENGTH_M
            ax.quiver(*translation, *vector, color=local_colors[axis_index], linewidth=1.4, arrow_length_ratio=0.15)
            ax.text(*(translation + vector*1.1), local_labels[axis_index], color=local_colors[axis_index], fontsize=8)

        # Build a visible plane patch around all polygon markers on this rigid plane.
        plane_marker_vertices = [
            np.asarray(marker.object_vertices_m, dtype=np.float64)
            for marker in rigid_plane.shape_markers
            if marker.num_sides != 0 and marker.object_vertices_m is not None
        ]

        if plane_marker_vertices:
            local_vertices = np.concatenate(plane_marker_vertices, axis=0)
            min_xy, max_xy = local_vertices.min(axis=0) - PLANE_MARGIN_M, local_vertices.max(axis=0) + PLANE_MARGIN_M
            patch_xy = np.array([
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]],
            ])
            patch_object = transformPlanePoints(patch_xy, rotation, translation)
            all_points.extend(patch_object)

            patch = Poly3DCollection([patch_object], alpha=0.10, edgecolor="0.45", linewidth=1)
            ax.add_collection3d(patch)

        # Transform and draw every marker from plane-local (x, y) into the common object frame.
        for marker_index, marker in enumerate(rigid_plane.shape_markers):
            if marker.num_sides == 0:
                print(f"Skipping circle marker P{plane_index}/M{marker_index}: circle visualization is not implemented yet")
                continue
            if marker.object_vertices_m is None:
                print(f"Skipping P{plane_index}/M{marker_index}: no object_vertices_m")
                continue

            vertices_object = transformPlanePoints(np.asarray(marker.object_vertices_m), rotation, translation)
            all_points.extend(vertices_object)
            closed = np.vstack((vertices_object, vertices_object[0]))

            b, g, r = COLOR_SPECS[marker.color_id].draw_bgr
            marker_color = (r/255.0, g/255.0, b/255.0)
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=marker_color, linewidth=3)
            ax.scatter(vertices_object[:, 0], vertices_object[:, 1], vertices_object[:, 2], color=[marker_color], s=35)

            center = vertices_object.mean(axis=0)
            ax.text(*center, f"P{plane_index} M{marker_index} {marker.color_id.name}", color=marker_color, fontsize=9)

            for vertex_index, vertex in enumerate(vertices_object):
                ax.text(*vertex, f" {vertex_index}", color=marker_color, fontsize=8)

    all_points = np.asarray(all_points)
    setAxesEqual(ax, all_points)

    ax.set_xlabel("Object x [m] — forward")
    ax.set_ylabel("Object y [m] — down")
    ax.set_zlabel("Object z [m] — right")
    ax.set_title(f"{object_type.name} vision model\np_object = R_object_from_plane @ [x, y, 0] + t_object_from_plane")
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an ObjectVisionSpec rigid-plane marker model.")
    parser.add_argument("--object", default=DEFAULT_OBJECT_TYPE.name, choices=[object_type.name for object_type in ObjectType])
    args = parser.parse_args()
    visualizeObjectVisionModel(ObjectType[args.object])