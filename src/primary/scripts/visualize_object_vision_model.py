import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.primary.object_vision_spec import (
    MODELS_DIR, OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec, ObjectVisionSpecId,
    RigidPlaneSpec, ShapeMarkerSpec, loadObjectVisionSpecModel, saveObjectVisionSpecModel,
)
from src.primary.color import COLOR_SPECS, ColorId


DEFAULT_OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.PAPER_PLANE_SHAPES_1
PLANE_MARGIN_M = 0.015
EMPTY_PLANE_HALF_SIZE_M = 0.03
OBJECT_AXIS_LENGTH_M = 0.08
PLANE_AXIS_LENGTH_M = 0.025

# Camera is on the +x, -y, +z side: front + top + left.
ISOMETRIC_ELEV_DEG = 35.26438968
ISOMETRIC_AZIM_DEG = -45.0
ISOMETRIC_ROLL_DEG = -120.0

# Orthographic-style presets are rolled so the airplane is oriented naturally:
# top/bottom keep the nose (+x) upward; front/back/left/right keep +y visually downward.
VIEW_PRESETS = {
    "Top": (0.0, -90.0, 90.0),       # camera on -y; +x up, +z left
    "Front": (0.0, 0.0, -90.0),      # camera on +x; +y down
    "Back": (0.0, 180.0, 90.0),      # camera on -x; +y down
    "Left": (90.0, -90.0, 180.0),    # camera on +z; +y down, nose left
    "Right": (-90.0, 90.0, 180.0),   # camera on -z; +y down, nose right
    "Bottom": (0.0, 90.0, -90.0),    # camera on +y; +x up
    "Isometric": (ISOMETRIC_ELEV_DEG, ISOMETRIC_AZIM_DEG, ISOMETRIC_ROLL_DEG),
}

UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
DEFAULT_DISPLAY_UNIT = "cm"


class EditableShape:
    def __init__(
        self, color_id: ColorId, vertices_xy_m: np.ndarray,
        minimum_contour_area_px: float | None = None, visible: bool = True,
    ):
        self.color_id = color_id
        self.object_vertices_m = np.asarray(vertices_xy_m, dtype=np.float64)
        self.minimum_contour_area_px = minimum_contour_area_px
        self.visible = visible

    @property
    def num_sides(self) -> int:
        return len(self.object_vertices_m)


class EditableRigidPlane:
    def __init__(
        self, rotation: np.ndarray, translation: np.ndarray,
        shape_markers: list[EditableShape] | None = None, visible: bool = True,
    ):
        self.rotation_object_from_plane = np.asarray(rotation, dtype=np.float64)
        self.translation_object_from_plane_m = np.asarray(translation, dtype=np.float64)
        self.shape_markers = shape_markers if shape_markers is not None else []
        self.visible = visible


class EdgePointSelection:
    def __init__(self, plane_index: int, shape_index: int, edge_index: int, t: float):
        self.plane_index = plane_index
        self.shape_index = shape_index
        self.edge_index = edge_index
        self.t = float(np.clip(t, 0.0, 1.0))


def copyModelFromSpec(object_vision_spec: ObjectVisionSpec) -> list[EditableRigidPlane]:
    rigid_planes = []

    for rigid_plane in object_vision_spec.rigid_planes:
        shapes = []

        for marker in rigid_plane.shape_markers:
            if marker.num_sides == 0 or marker.object_vertices_m is None:
                continue

            shapes.append(EditableShape(
                marker.color_id, np.asarray(marker.object_vertices_m, dtype=np.float64).copy(),
                getattr(marker, "minimum_contour_area_px", None),
            ))

        rigid_planes.append(EditableRigidPlane(
            rigid_plane.rotation_object_from_plane.copy(),
            rigid_plane.translation_object_from_plane_m.copy(),
            shapes,
        ))

    return rigid_planes


def transformPlanePoints(points_xy: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points_plane = np.column_stack((points_xy, np.zeros(len(points_xy))))
    return (rotation@points_plane.T).T + translation


def getEdgePointObjectPosition(rigid_planes: list[EditableRigidPlane], selection: EdgePointSelection) -> np.ndarray | None:
    try:
        plane = rigid_planes[selection.plane_index]
        shape = plane.shape_markers[selection.shape_index]
        vertex_a = shape.object_vertices_m[selection.edge_index]
        vertex_b = shape.object_vertices_m[(selection.edge_index + 1)%shape.num_sides]
    except IndexError:
        return None

    point_xy = (1.0 - selection.t)*vertex_a + selection.t*vertex_b
    return transformPlanePoints(
        np.asarray([point_xy]), plane.rotation_object_from_plane, plane.translation_object_from_plane_m,
    )[0]


def edgeSelectionName(selection: EdgePointSelection, rigid_planes: list[EditableRigidPlane]) -> str:
    shape = rigid_planes[selection.plane_index].shape_markers[selection.shape_index]
    next_vertex = (selection.edge_index + 1)%shape.num_sides
    return f"P{selection.plane_index} S{selection.shape_index} E{selection.edge_index} (V{selection.edge_index}→V{next_vertex})"


def setAxesEqual(ax, points: np.ndarray, zoom_factor: float = 1.0) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs)/2.0
    base_radius = max(np.max(maxs - mins)/2.0, 0.01)
    radius = base_radius/max(zoom_factor, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def drawModel(
    ax, rigid_planes: list[EditableRigidPlane], object_type: ObjectType,
    view_elev: float = ISOMETRIC_ELEV_DEG, view_azim: float = ISOMETRIC_AZIM_DEG, view_roll: float = 0.0,
    edge_pick_data: list | None = None, measurement_points: list[EdgePointSelection] | None = None,
    display_unit: str = DEFAULT_DISPLAY_UNIT, zoom_factor: float = 1.0,
) -> None:
    ax.clear()
    all_points = [np.zeros(3)]
    measurement_points = measurement_points if measurement_points is not None else []

    if edge_pick_data is not None:
        edge_pick_data.clear()

    # Object/reference frame: +x forward, +y down, +z left.
    for vector, label, color in zip(
        np.eye(3)*OBJECT_AXIS_LENGTH_M,
        ("+x forward", "+y down", "+z left"),
        ("r", "g", "b"),
    ):
        ax.quiver(0, 0, 0, *vector, color=color, linewidth=2, arrow_length_ratio=0.12)
        ax.text(*(vector*1.08), label, color=color)

    ax.scatter([0], [0], [0], color="k", s=35)
    ax.text(0, 0, 0, " object origin")

    for plane_index, rigid_plane in enumerate(rigid_planes):
        rotation = rigid_plane.rotation_object_from_plane
        translation = rigid_plane.translation_object_from_plane_m

        # Keep all planes in the axis limits even when hidden so visibility toggles do not change zoom.
        if rigid_plane.shape_markers:
            local_vertices = np.concatenate([shape.object_vertices_m for shape in rigid_plane.shape_markers], axis=0)
            min_xy, max_xy = local_vertices.min(axis=0) - PLANE_MARGIN_M, local_vertices.max(axis=0) + PLANE_MARGIN_M
        else:
            min_xy = np.array([-EMPTY_PLANE_HALF_SIZE_M, -EMPTY_PLANE_HALF_SIZE_M])
            max_xy = np.array([EMPTY_PLANE_HALF_SIZE_M, EMPTY_PLANE_HALF_SIZE_M])

        patch_xy = np.array([
            [min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]],
        ])
        patch_object = transformPlanePoints(patch_xy, rotation, translation)
        all_points.extend(patch_object)

        if rigid_plane.visible:
            # Plane-local +x, +y, and normal transformed into the object frame.
            local_axes_object = rotation@np.eye(3)

            for axis_index, (label, color) in enumerate(zip(
                (f"P{plane_index} +x", f"P{plane_index} +y", f"P{plane_index} normal"),
                ("tab:red", "tab:green", "tab:blue"),
            )):
                vector = local_axes_object[:, axis_index]*PLANE_AXIS_LENGTH_M
                ax.quiver(*translation, *vector, color=color, linewidth=1.4, arrow_length_ratio=0.15)
                ax.text(*(translation + vector*1.1), label, color=color, fontsize=8)

            ax.add_collection3d(Poly3DCollection([patch_object], alpha=0.10, edgecolor="0.45", linewidth=1))

        # Shapes have their own visibility; hiding a plane surface does not hide its markers.
        for shape_index, shape in enumerate(rigid_plane.shape_markers):
            vertices_object = transformPlanePoints(shape.object_vertices_m, rotation, translation)
            all_points.extend(vertices_object)

            if not shape.visible:
                continue

            b, g, r = COLOR_SPECS[shape.color_id].draw_bgr
            marker_color = (r/255.0, g/255.0, b/255.0)

            # Draw each edge separately so clicks can select a continuous point on that exact edge.
            for edge_index in range(shape.num_sides):
                point_a = vertices_object[edge_index]
                point_b = vertices_object[(edge_index + 1)%shape.num_sides]
                ax.plot(
                    [point_a[0], point_b[0]], [point_a[1], point_b[1]], [point_a[2], point_b[2]],
                    color=marker_color, linewidth=3,
                )

                if edge_pick_data is not None:
                    edge_pick_data.append((
                        plane_index, shape_index, edge_index, point_a.copy(), point_b.copy(),
                    ))

            ax.scatter(vertices_object[:, 0], vertices_object[:, 1], vertices_object[:, 2], color=[marker_color], s=42)

            center = vertices_object.mean(axis=0)
            ax.text(*center, f"P{plane_index} S{shape_index} {shape.color_id.name}", color=marker_color, fontsize=9)

            for vertex_index, vertex in enumerate(vertices_object):
                ax.text(*vertex, f" {vertex_index}", color=marker_color, fontsize=8)

    # Show the two continuously movable measurement points and their connecting segment.
    measurement_positions = []

    for index, selection in enumerate(measurement_points[:2]):
        point = getEdgePointObjectPosition(rigid_planes, selection)
        if point is None:
            continue

        measurement_positions.append(point)
        shape_visible = rigid_planes[selection.plane_index].shape_markers[selection.shape_index].visible

        if shape_visible:
            label = "A" if index == 0 else "B"
            ax.scatter([point[0]], [point[1]], [point[2]], s=145, facecolors="none", edgecolors="black", linewidths=2)
            ax.text(*point, f"  {label}", color="black", fontsize=10, fontweight="bold")

    if len(measurement_positions) == 2:
        point_a, point_b = measurement_positions
        shape_a = rigid_planes[measurement_points[0].plane_index].shape_markers[measurement_points[0].shape_index]
        shape_b = rigid_planes[measurement_points[1].plane_index].shape_markers[measurement_points[1].shape_index]
        if shape_a.visible or shape_b.visible:
            ax.plot(
                [point_a[0], point_b[0]], [point_a[1], point_b[1]], [point_a[2], point_b[2]],
                color="black", linestyle="--", linewidth=1.5,
            )

    setAxesEqual(ax, np.asarray(all_points), zoom_factor)
    unit_to_m = UNIT_TO_METERS[display_unit]
    tick_formatter = FuncFormatter(lambda value, _position: f"{value/unit_to_m:g}")
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_formatter(tick_formatter)
    ax.zaxis.set_major_formatter(tick_formatter)
    ax.set_xlabel(f"Object x [{display_unit}] — forward")
    ax.set_ylabel(f"Object y [{display_unit}] — down")
    ax.set_zlabel(f"Object z [{display_unit}] — left")
    ax.set_title(f"{object_type.name}\np_object = R @ [x, y, 0] + t")

    try:
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
    except TypeError:
        ax.view_init(elev=view_elev, azim=view_azim)


def rotationMatrixFromAnglesDeg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    # Apply local x rotation, then y, then z: R = Rz @ Ry @ Rx.
    x, y, z = np.deg2rad([x_deg, y_deg, z_deg])
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz@ry@rx


def anglesDegFromRotationMatrix(rotation: np.ndarray) -> np.ndarray:
    # Inverse of R = Rz @ Ry @ Rx. Near gimbal lock, z is chosen as 0.
    sy = -float(rotation[2, 0])
    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    cy = np.cos(y)

    if abs(cy) > 1e-8:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        z = 0.0

    return np.rad2deg([x, y, z])


def parseRotation(entries: list[list[tk.Entry]]) -> np.ndarray:
    return np.array([[float(entries[row][col].get()) for col in range(3)] for row in range(3)], dtype=np.float64)


def parseTranslation(entries: list[tk.Entry], unit_to_m: float = 1.0) -> np.ndarray:
    return unit_to_m*np.array([float(entry.get()) for entry in entries], dtype=np.float64)


def parsePoints(text: str, unit_to_m: float = 1.0) -> np.ndarray:
    points = []

    for line in text.replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue

        values = [float(value) for value in line.replace("[", " ").replace("]", " ").replace(",", " ").split()]
        if len(values) != 2:
            raise ValueError("Each point must contain exactly x y")
        points.append(values)

    if len(points) < 3:
        raise ValueError("A polygon requires at least 3 points")

    return unit_to_m*np.asarray(points, dtype=np.float64)


def createObjectVisionSpec(source_spec: ObjectVisionSpec, rigid_planes: list[EditableRigidPlane]) -> ObjectVisionSpec:
    color_ids = []

    for plane in rigid_planes:
        for shape in plane.shape_markers:
            if shape.color_id not in color_ids:
                color_ids.append(shape.color_id)

    if not color_ids:
        color_ids = list(source_spec.color_ids)

    return ObjectVisionSpec(
        object_type=source_spec.object_type,
        color_ids=color_ids,
        minimum_contour_area_px=source_spec.minimum_contour_area_px,
        polygon_epsilon_ratio=source_spec.polygon_epsilon_ratio,
        shape_group_distance_factor=source_spec.shape_group_distance_factor,
        rigid_planes=[
            RigidPlaneSpec(
                rotation_object_from_plane=plane.rotation_object_from_plane.copy(),
                translation_object_from_plane_m=plane.translation_object_from_plane_m.copy(),
                shape_markers=[
                    ShapeMarkerSpec(
                        color_id=shape.color_id, num_sides=shape.num_sides,
                        object_vertices_m=shape.object_vertices_m.copy(),
                        minimum_contour_area_px=shape.minimum_contour_area_px,
                    )
                    for shape in plane.shape_markers
                ],
            )
            for plane in rigid_planes
        ],
        width=source_spec.width, height=source_spec.height, length=source_spec.length,
    )


def objectVisionSpecCode(spec_id_name: str, spec: ObjectVisionSpec) -> str:
    lines = [
        "import numpy as np",
        "",
        "from src.primary.color import ColorId",
        "from src.primary.object_vision_spec import ObjectType, ObjectVisionSpec, RigidPlaneSpec, ShapeMarkerSpec",
        "",
        f"# Suggested ObjectVisionSpecId member: {spec_id_name} = auto()",
        f'EXPORTED_OBJECT_VISION_SPEC_ID_NAME = "{spec_id_name}"',
        "",
        "EXPORTED_OBJECT_VISION_SPEC = ObjectVisionSpec(",
        f"    object_type=ObjectType.{spec.object_type.name},",
        "    color_ids=[" + ", ".join(f"ColorId.{color_id.name}" for color_id in spec.color_ids) + "],",
        f"    minimum_contour_area_px={spec.minimum_contour_area_px!r},",
        f"    polygon_epsilon_ratio={spec.polygon_epsilon_ratio!r},",
        f"    shape_group_distance_factor={spec.shape_group_distance_factor!r},",
        "    rigid_planes=[",
    ]

    for plane in spec.rigid_planes:
        lines += [
            "        RigidPlaneSpec(",
            "            rotation_object_from_plane=np.array([",
            *[f"                {row.tolist()}," for row in plane.rotation_object_from_plane],
            "            ], dtype=np.float64),",
            f"            translation_object_from_plane_m=np.array({plane.translation_object_from_plane_m.tolist()}, dtype=np.float64),",
            "            shape_markers=[",
        ]

        for marker in plane.shape_markers:
            lines += [
                "                ShapeMarkerSpec(",
                f"                    color_id=ColorId.{marker.color_id.name}, num_sides={marker.num_sides},",
                f"                    object_vertices_m=np.array({marker.object_vertices_m.tolist()}, dtype=np.float64),",
            ]
            if marker.minimum_contour_area_px is not None:
                lines.append(f"                    minimum_contour_area_px={marker.minimum_contour_area_px!r},")
            lines.append("                ),")

        lines += ["            ],", "        ),"]

    lines += [
        "    ],",
        f"    width={spec.width!r}, height={spec.height!r}, length={spec.length!r},",
        ")",
        "",
        f"# Registry entry after adding the enum member:",
        f"# ObjectVisionSpecId.{spec_id_name}: EXPORTED_OBJECT_VISION_SPEC,",
        "",
    ]
    return "\n".join(lines)


class ModelEditor(tk.Tk):
    def __init__(self, object_vision_spec_id: ObjectVisionSpecId):
        super().__init__()
        self.title("Object Vision Model Editor")
        self.geometry("1500x940")
        self.minsize(1150, 760)

        self.object_vision_spec_id: ObjectVisionSpecId | None = object_vision_spec_id
        self.source_spec = OBJECT_VISION_SPECS[object_vision_spec_id]
        self.model_name = object_vision_spec_id.name
        self.model_path: Path | None = None
        self.rigid_planes = copyModelFromSpec(self.source_spec)
        self.selected_plane_index: int | None = None
        self.selected_shape_index: int | None = None

        self.edge_pick_data: list = []
        self.measurement_points: list[EdgePointSelection] = []
        self.mouse_press_xy: tuple[float, float] | None = None
        self.updating_measure_controls = False

        self.display_unit = DEFAULT_DISPLAY_UNIT
        self.zoom_factor = 1.0

        # Default view: front + top + left.
        self.view_elev = ISOMETRIC_ELEV_DEG
        self.view_azim = ISOMETRIC_AZIM_DEG
        self.view_roll = ISOMETRIC_ROLL_DEG

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.controls = ttk.Frame(self, padding=10)
        self.controls.grid(row=0, column=0, sticky="nsew")
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        self.buildControls()
        self.buildPlot()
        self.updateUnitLabels()
        self.refreshPlaneList()
        self.redraw()

    def buildControls(self) -> None:
        unit_frame = ttk.Frame(self.controls)
        unit_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Label(unit_frame, text="Display/edit units:").pack(side="left")
        self.unit_var = tk.StringVar(value=self.display_unit)
        self.unit_combo = ttk.Combobox(
            unit_frame, textvariable=self.unit_var, values=list(UNIT_TO_METERS), state="readonly", width=6,
        )
        self.unit_combo.pack(side="left", padx=(6, 0))
        self.unit_combo.bind("<<ComboboxSelected>>", self.onUnitChanged)

        ttk.Label(self.controls, text="Rigid planes").grid(row=1, column=0, columnspan=3, sticky="w")
        self.plane_list = tk.Listbox(self.controls, height=6, exportselection=False)
        self.plane_list.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(3, 5))
        self.plane_list.bind("<<ListboxSelect>>", self.onPlaneSelected)

        ttk.Button(self.controls, text="Add plane", command=self.addPlane).grid(row=3, column=0, sticky="ew")
        ttk.Button(self.controls, text="Delete plane", command=self.deletePlane).grid(row=3, column=1, sticky="ew")
        ttk.Button(self.controls, text="Apply plane", command=self.applyPlane).grid(row=3, column=2, sticky="ew")

        self.plane_visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.controls, text="Selected plane surface visible", variable=self.plane_visible_var,
            command=self.onPlaneVisibilityChanged,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))

        ttk.Label(self.controls, text="Rotation angles [deg]").grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Label(self.controls, text="Convention: R = Rz(z) @ Ry(y) @ Rx(x)").grid(row=6, column=0, columnspan=3, sticky="w")

        self.angle_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.angle_entries, self.angle_scales = [], []
        self.updating_rotation_controls = False

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            row = 7 + axis_index
            ttk.Label(self.controls, text=f"{axis_name}:").grid(row=row, column=0, sticky="w")

            entry = ttk.Entry(self.controls, width=9)
            entry.grid(row=row, column=1, padx=(2, 4), sticky="ew")
            entry.bind("<Return>", lambda _event, i=axis_index: self.onAngleEntryChanged(i))
            entry.bind("<FocusOut>", lambda _event, i=axis_index: self.onAngleEntryChanged(i))
            self.angle_entries.append(entry)

            scale = ttk.Scale(
                self.controls, from_=-180.0, to=180.0, variable=self.angle_vars[axis_index],
                command=lambda _value, i=axis_index: self.onAngleSliderChanged(i),
            )
            scale.grid(row=row, column=2, sticky="ew")
            self.angle_scales.append(scale)

        ttk.Label(self.controls, text="Rotation matrix R").grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.rotation_entries = []

        for row in range(3):
            entry_row = []
            for col in range(3):
                entry = ttk.Entry(self.controls, width=10)
                entry.grid(row=11 + row, column=col, padx=2, pady=2, sticky="ew")
                entry_row.append(entry)
            self.rotation_entries.append(entry_row)

        self.translation_label_var = tk.StringVar()
        ttk.Label(self.controls, textvariable=self.translation_label_var).grid(row=14, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.translation_entries = []

        for col, axis_name in enumerate(("x", "y", "z")):
            frame = ttk.Frame(self.controls)
            frame.grid(row=15, column=col, padx=2, sticky="ew")
            ttk.Label(frame, text=axis_name).pack(side="left")
            entry = ttk.Entry(frame, width=9)
            entry.pack(side="left", fill="x", expand=True)
            self.translation_entries.append(entry)

        ttk.Separator(self.controls).grid(row=16, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(self.controls, text="Shapes on selected plane").grid(row=17, column=0, columnspan=3, sticky="w")
        self.shape_list = tk.Listbox(self.controls, height=5, exportselection=False)
        self.shape_list.grid(row=18, column=0, columnspan=3, sticky="ew", pady=(3, 5))
        self.shape_list.bind("<<ListboxSelect>>", self.onShapeSelected)

        ttk.Button(self.controls, text="Add shape", command=self.addShape).grid(row=19, column=0, sticky="ew")
        ttk.Button(self.controls, text="Delete shape", command=self.deleteShape).grid(row=19, column=1, sticky="ew")
        ttk.Button(self.controls, text="Apply shape", command=self.applyShape).grid(row=19, column=2, sticky="ew")

        self.shape_visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.controls, text="Selected shape visible", variable=self.shape_visible_var,
            command=self.onShapeVisibilityChanged,
        ).grid(row=20, column=0, columnspan=3, sticky="w", pady=(5, 0))

        ttk.Label(self.controls, text="Color").grid(row=21, column=0, sticky="w", pady=(6, 2))
        self.color_var = tk.StringVar(value=next(iter(ColorId.__members__)))
        self.color_combo = ttk.Combobox(self.controls, textvariable=self.color_var, values=list(ColorId.__members__), state="readonly")
        self.color_combo.grid(row=21, column=1, columnspan=2, sticky="ew", pady=(6, 2))

        self.points_label_var = tk.StringVar()
        ttk.Label(self.controls, textvariable=self.points_label_var).grid(row=22, column=0, columnspan=3, sticky="w", pady=(4, 2))
        self.points_text = tk.Text(self.controls, width=36, height=6)
        self.points_text.grid(row=23, column=0, columnspan=3, sticky="ew")

        ttk.Button(self.controls, text="Show / copy spec code", command=self.showCode).grid(row=24, column=0, columnspan=3, sticky="ew", pady=(6, 2))

        ttk.Separator(self.controls).grid(row=25, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(self.controls, text="Model / ObjectVisionSpec").grid(row=26, column=0, columnspan=3, sticky="w")

        self.spec_id_var = tk.StringVar(value=self.object_vision_spec_id.name)
        self.spec_id_combo = ttk.Combobox(
            self.controls, textvariable=self.spec_id_var,
            values=[spec_id.name for spec_id in ObjectVisionSpecId], state="readonly",
        )
        self.spec_id_combo.grid(row=27, column=0, columnspan=2, sticky="ew")
        ttk.Button(self.controls, text="Load registered", command=self.loadRegisteredSpec).grid(row=27, column=2, sticky="ew")

        self.model_name_var = tk.StringVar(value=f"Current: {self.model_name}")
        ttk.Label(self.controls, textvariable=self.model_name_var).grid(row=28, column=0, columnspan=3, sticky="w", pady=(4, 2))

        ttk.Button(self.controls, text="New blank", command=self.newBlankModel).grid(row=29, column=0, sticky="ew")
        ttk.Button(self.controls, text="Import JSON", command=self.importModelJson).grid(row=29, column=1, sticky="ew")
        ttk.Button(self.controls, text="Save", command=self.saveModelJson).grid(row=29, column=2, sticky="ew")

        ttk.Button(self.controls, text="Save As JSON", command=self.saveModelJsonAs).grid(row=30, column=0, columnspan=2, sticky="ew")
        ttk.Button(self.controls, text="Export Python", command=self.exportPythonSpec).grid(row=30, column=2, sticky="ew")

        for col in range(3):
            self.controls.columnconfigure(col, weight=1)

    def buildPlot(self) -> None:
        self.figure = plt.Figure(figsize=(9, 8))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.setPlotView()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.onPlotMousePress)
        self.canvas.mpl_connect("button_release_event", self.onPlotMouseRelease)

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        view_frame = ttk.LabelFrame(self.plot_frame, text="View presets", padding=5)
        view_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 5))

        for column, name in enumerate(("Top", "Front", "Back", "Left", "Right", "Bottom", "Isometric")):
            ttk.Button(view_frame, text=name, command=lambda n=name: self.setViewPreset(n)).grid(row=0, column=column, padx=2, sticky="ew")
            view_frame.columnconfigure(column, weight=1)

        ttk.Label(view_frame, text="Zoom").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.zoom_var = tk.DoubleVar(value=self.zoom_factor)
        self.zoom_scale = ttk.Scale(
            view_frame, from_=0.6, to=3.0, variable=self.zoom_var, command=self.onZoomChanged,
        )
        self.zoom_scale.grid(row=1, column=1, columnspan=5, sticky="ew", padx=6, pady=(5, 0))
        self.zoom_value_var = tk.StringVar(value=f"{self.zoom_factor:.2f}×")
        ttk.Label(view_frame, textvariable=self.zoom_value_var, width=8).grid(row=1, column=6, sticky="e", pady=(5, 0))

        measurement_frame = ttk.LabelFrame(self.plot_frame, text="Edge-to-edge measurement", padding=6)
        measurement_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        measurement_frame.columnconfigure(1, weight=1)

        self.measure_a_var = tk.StringVar(value="A: click anywhere on a shape edge")
        self.measure_b_var = tk.StringVar(value="B: click anywhere on a second edge")
        self.measure_delta_var = tk.StringVar(value="Δ(B-A): —")
        self.measure_axis_abs_var = tk.StringVar(value="|Δ axes|: —")
        self.measure_distance_var = tk.StringVar(value="3D distance: —")
        self.measure_t_vars = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        self.measure_t_value_vars = [tk.StringVar(value="t = —"), tk.StringVar(value="t = —")]
        self.measure_coord_entries: list[list[ttk.Entry]] = [[], []]

        ttk.Label(measurement_frame, textvariable=self.measure_a_var).grid(row=0, column=0, columnspan=5, sticky="w")
        ttk.Label(measurement_frame, text="Slide A along edge").grid(row=1, column=0, sticky="w")
        self.measure_a_scale = ttk.Scale(
            measurement_frame, from_=0.0, to=1.0, variable=self.measure_t_vars[0],
            command=lambda value: self.onMeasurementSlider(0, value),
        )
        self.measure_a_scale.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Label(measurement_frame, textvariable=self.measure_t_value_vars[0], width=10).grid(row=1, column=4, sticky="e")
        self.measure_a_xyz_label_var = tk.StringVar()
        ttk.Label(measurement_frame, textvariable=self.measure_a_xyz_label_var).grid(row=2, column=0, sticky="w")

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            frame = ttk.Frame(measurement_frame)
            frame.grid(row=2, column=axis_index + 1, padx=2, sticky="ew")
            ttk.Label(frame, text=axis_name).pack(side="left")
            entry = ttk.Entry(frame, width=11)
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<Return>", lambda _event, i=0, a=axis_index: self.onMeasurementCoordinateEntered(i, a))
            self.measure_coord_entries[0].append(entry)

        ttk.Label(measurement_frame, textvariable=self.measure_b_var).grid(row=3, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(measurement_frame, text="Slide B along edge").grid(row=4, column=0, sticky="w")
        self.measure_b_scale = ttk.Scale(
            measurement_frame, from_=0.0, to=1.0, variable=self.measure_t_vars[1],
            command=lambda value: self.onMeasurementSlider(1, value),
        )
        self.measure_b_scale.grid(row=4, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Label(measurement_frame, textvariable=self.measure_t_value_vars[1], width=10).grid(row=4, column=4, sticky="e")
        self.measure_b_xyz_label_var = tk.StringVar()
        ttk.Label(measurement_frame, textvariable=self.measure_b_xyz_label_var).grid(row=5, column=0, sticky="w")

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            frame = ttk.Frame(measurement_frame)
            frame.grid(row=5, column=axis_index + 1, padx=2, sticky="ew")
            ttk.Label(frame, text=axis_name).pack(side="left")
            entry = ttk.Entry(frame, width=11)
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<Return>", lambda _event, i=1, a=axis_index: self.onMeasurementCoordinateEntered(i, a))
            self.measure_coord_entries[1].append(entry)

        ttk.Label(measurement_frame, textvariable=self.measure_delta_var).grid(row=6, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(measurement_frame, textvariable=self.measure_axis_abs_var).grid(row=7, column=0, columnspan=5, sticky="w")
        ttk.Label(measurement_frame, textvariable=self.measure_distance_var).grid(row=8, column=0, columnspan=4, sticky="w")
        ttk.Button(measurement_frame, text="Clear", command=self.clearMeasurement).grid(row=8, column=4, sticky="e")

        for column in range(1, 4):
            measurement_frame.columnconfigure(column, weight=1)

        self.updateMeasurementControls()

    def unitToMeters(self) -> float:
        return UNIT_TO_METERS[self.display_unit]

    def metersToDisplay(self, value):
        return np.asarray(value)/self.unitToMeters()

    def updateUnitLabels(self) -> None:
        unit = self.display_unit
        if hasattr(self, "translation_label_var"):
            self.translation_label_var.set(f"Translation t [{unit}]")
        if hasattr(self, "points_label_var"):
            self.points_label_var.set(f"Plane-local points [{unit}] — one x y pair per line")
        if hasattr(self, "measure_a_xyz_label_var"):
            self.measure_a_xyz_label_var.set(f"A xyz [{unit}] (type one + Enter):")
            self.measure_b_xyz_label_var.set(f"B xyz [{unit}] (type one + Enter):")

    def onUnitChanged(self, _event=None) -> None:
        self.display_unit = self.unit_var.get()
        self.updateUnitLabels()

        if self.selected_plane_index is not None:
            self.loadPlaneFields()
        if self.selected_shape_index is not None:
            self.loadShapeFields()

        self.redraw()

    def onZoomChanged(self, value: str) -> None:
        self.zoom_factor = float(value)
        self.zoom_value_var.set(f"{self.zoom_factor:.2f}×")
        self.redraw()

    def currentObjectVisionSpec(self) -> ObjectVisionSpec:
        return createObjectVisionSpec(self.source_spec, self.rigid_planes)

    def setModel(self, spec: ObjectVisionSpec, name: str, spec_id: ObjectVisionSpecId | None = None, path: Path | None = None) -> None:
        self.object_vision_spec_id = spec_id
        self.source_spec = spec
        self.model_name = name
        self.model_path = path
        self.rigid_planes = copyModelFromSpec(spec)
        self.selected_plane_index = None
        self.selected_shape_index = None
        self.measurement_points.clear()
        self.model_name_var.set(f"Current: {name}")
        if spec_id is not None:
            self.spec_id_var.set(spec_id.name)
        self.refreshPlaneList(0 if self.rigid_planes else None)
        self.redraw()

    def loadRegisteredSpec(self) -> None:
        spec_id = ObjectVisionSpecId[self.spec_id_var.get()]
        self.setModel(OBJECT_VISION_SPECS[spec_id], spec_id.name, spec_id=spec_id)

    def newBlankModel(self) -> None:
        name = simpledialog.askstring("New model", "Model name:", initialvalue=f"{self.source_spec.object_type.name.lower()}_new", parent=self)
        if not name:
            return

        blank_spec = ObjectVisionSpec(
            object_type=self.source_spec.object_type,
            color_ids=list(self.source_spec.color_ids),
            minimum_contour_area_px=self.source_spec.minimum_contour_area_px,
            polygon_epsilon_ratio=self.source_spec.polygon_epsilon_ratio,
            shape_group_distance_factor=self.source_spec.shape_group_distance_factor,
            rigid_planes=[],
            width=self.source_spec.width, height=self.source_spec.height, length=self.source_spec.length,
        )
        self.setModel(blank_spec, name)

    def importModelJson(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path_text = filedialog.askopenfilename(
            title="Import ObjectVisionSpec model",
            initialdir=MODELS_DIR,
            filetypes=[("ObjectVisionSpec JSON", "*.json"), ("All files", "*.*")],
        )
        if not path_text:
            return

        try:
            path = Path(path_text)
            spec = loadObjectVisionSpecModel(path)
        except Exception as error:
            messagebox.showerror("Import failed", str(error))
            return

        self.setModel(spec, path.stem, path=path)

    def saveModelJson(self) -> None:
        if self.model_path is None:
            self.saveModelJsonAs()
            return

        try:
            saveObjectVisionSpecModel(self.currentObjectVisionSpec(), self.model_path)
        except Exception as error:
            messagebox.showerror("Save failed", str(error))
            return

        self.model_name_var.set(f"Current: {self.model_path.stem}")
        messagebox.showinfo("Saved", f"Saved model to:\n{self.model_path}")

    def saveModelJsonAs(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        initial_name = (self.model_path.stem if self.model_path is not None else self.model_name).lower() + ".json"
        path_text = filedialog.asksaveasfilename(
            title="Save ObjectVisionSpec model",
            initialdir=MODELS_DIR,
            initialfile=initial_name,
            defaultextension=".json",
            filetypes=[("ObjectVisionSpec JSON", "*.json"), ("All files", "*.*")],
        )
        if not path_text:
            return

        self.model_path = Path(path_text)
        self.model_name = self.model_path.stem
        self.saveModelJson()

    def exportPythonSpec(self) -> None:
        spec_id_name = simpledialog.askstring(
            "Export Python spec",
            "Suggested ObjectVisionSpecId name:",
            initialvalue=(self.object_vision_spec_id.name if self.object_vision_spec_id is not None else self.model_name.upper()),
            parent=self,
        )
        if not spec_id_name:
            return

        spec_id_name = spec_id_name.strip().upper().replace(" ", "_")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path_text = filedialog.asksaveasfilename(
            title="Export Python ObjectVisionSpec",
            initialdir=MODELS_DIR,
            initialfile=spec_id_name.lower() + ".py",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not path_text:
            return

        try:
            Path(path_text).write_text(objectVisionSpecCode(spec_id_name, self.currentObjectVisionSpec()), encoding="utf-8")
        except Exception as error:
            messagebox.showerror("Export failed", str(error))
            return

        messagebox.showinfo(
            "Exported",
            "Exported a standalone Python spec file.\n"
            "When you promote it into detection, add the suggested member to ObjectVisionSpecId "
            "and add the spec to OBJECT_VISION_SPECS.",
        )

    def savePlotView(self) -> None:
        self.view_elev = float(self.ax.elev)
        self.view_azim = float(self.ax.azim)
        self.view_roll = float(getattr(self.ax, "roll", 0.0))

    def setPlotView(self) -> None:
        try:
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim, roll=self.view_roll)
        except TypeError:
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim)

    def setViewPreset(self, name: str) -> None:
        self.view_elev, self.view_azim, self.view_roll = VIEW_PRESETS[name]
        self.setPlotView()
        self.canvas.draw_idle()

    def onPlotMousePress(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1:
            self.mouse_press_xy = None
            return

        self.mouse_press_xy = (event.x, event.y)

    def onPlotMouseRelease(self, event) -> None:
        self.savePlotView()

        if self.mouse_press_xy is None or event.inaxes != self.ax or event.button != 1:
            self.mouse_press_xy = None
            return

        press_x, press_y = self.mouse_press_xy
        self.mouse_press_xy = None

        # A drag rotates the 3D camera; only a nearly stationary click selects an edge point.
        if np.hypot(event.x - press_x, event.y - press_y) > 5.0:
            return

        if getattr(self.toolbar, "mode", ""):
            return

        nearest = self.findNearestEdgePoint(event.x, event.y, maximum_distance_px=10.0)
        if nearest is None:
            return

        plane_index, shape_index, edge_index, t = nearest
        selection = EdgePointSelection(plane_index, shape_index, edge_index, t)

        if len(self.measurement_points) >= 2:
            self.measurement_points = [selection]
        else:
            self.measurement_points.append(selection)

        self.updateMeasurementControls()
        self.redraw()

    def findNearestEdgePoint(self, mouse_x: float, mouse_y: float, maximum_distance_px: float) -> tuple | None:
        projection = self.ax.get_proj()
        best_result, best_distance = None, float("inf")

        for plane_index, shape_index, edge_index, point_a, point_b in self.edge_pick_data:
            x1, y1, _ = proj3d.proj_transform(*point_a, projection)
            x2, y2, _ = proj3d.proj_transform(*point_b, projection)
            pixel_a = self.ax.transData.transform((x1, y1))
            pixel_b = self.ax.transData.transform((x2, y2))
            segment = pixel_b - pixel_a
            length_squared = float(segment@segment)

            if length_squared <= 1e-12:
                continue

            t = float(np.clip(((np.array([mouse_x, mouse_y]) - pixel_a)@segment)/length_squared, 0.0, 1.0))
            nearest_pixel = pixel_a + t*segment
            distance = float(np.linalg.norm(np.array([mouse_x, mouse_y]) - nearest_pixel))

            if distance < best_distance:
                best_distance = distance
                best_result = (plane_index, shape_index, edge_index, t)

        return best_result if best_distance <= maximum_distance_px else None

    def redraw(self) -> None:
        self.savePlotView()
        self.validateMeasurementPoints()
        drawModel(
            self.ax, self.rigid_planes, self.source_spec.object_type,
            self.view_elev, self.view_azim, self.view_roll,
            self.edge_pick_data, self.measurement_points,
            self.display_unit, self.zoom_factor,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.updateMeasurementControls()

    def validateMeasurementPoints(self) -> None:
        self.measurement_points = [
            selection for selection in self.measurement_points
            if getEdgePointObjectPosition(self.rigid_planes, selection) is not None
        ][:2]

    def getMeasurementEdgeLengthM(self, selection: EdgePointSelection) -> float:
        point_0 = getEdgePointObjectPosition(
            self.rigid_planes,
            EdgePointSelection(selection.plane_index, selection.shape_index, selection.edge_index, 0.0),
        )
        point_1 = getEdgePointObjectPosition(
            self.rigid_planes,
            EdgePointSelection(selection.plane_index, selection.shape_index, selection.edge_index, 1.0),
        )
        return float(np.linalg.norm(point_1 - point_0))

    def updateMeasurementControls(self) -> None:
        if not hasattr(self, "measure_a_scale"):
            return

        self.updating_measure_controls = True
        scales = [self.measure_a_scale, self.measure_b_scale]

        for index, scale in enumerate(scales):
            if index < len(self.measurement_points):
                selection = self.measurement_points[index]
                edge_length_display = self.getMeasurementEdgeLengthM(selection)/self.unitToMeters()
                distance_display = selection.t*edge_length_display
                scale.configure(from_=0.0, to=max(edge_length_display, 1e-12))
                self.measure_t_vars[index].set(distance_display)
                self.measure_t_value_vars[index].set(f"s = {distance_display:.3f} {self.display_unit}")
                scale.state(["!disabled"])
            else:
                scale.configure(from_=0.0, to=1.0)
                self.measure_t_vars[index].set(0.0)
                self.measure_t_value_vars[index].set("s = —")
                scale.state(["disabled"])

        for point_index, entries in enumerate(self.measure_coord_entries):
            if point_index < len(self.measurement_points):
                point = self.metersToDisplay(
                    getEdgePointObjectPosition(self.rigid_planes, self.measurement_points[point_index])
                )
                for axis_index, entry in enumerate(entries):
                    entry.state(["!disabled"])
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{point[axis_index]:.6f}")
            else:
                for entry in entries:
                    entry.delete(0, tk.END)
                    entry.state(["disabled"])

        self.updating_measure_controls = False

        if not self.measurement_points:
            self.measure_a_var.set("A: click anywhere on a shape edge")
            self.measure_b_var.set("B: click anywhere on a second edge")
            self.measure_delta_var.set("Δ(B-A): —")
            self.measure_axis_abs_var.set("|Δ axes|: —")
            self.measure_distance_var.set("3D distance: —")
            return

        point_a_m = getEdgePointObjectPosition(self.rigid_planes, self.measurement_points[0])
        point_a = self.metersToDisplay(point_a_m)
        self.measure_a_var.set(
            f"A: {edgeSelectionName(self.measurement_points[0], self.rigid_planes)}, "
            f"({point_a[0]:.4f}, {point_a[1]:.4f}, {point_a[2]:.4f}) {self.display_unit}"
        )

        if len(self.measurement_points) < 2:
            self.measure_b_var.set("B: click anywhere on a second edge")
            self.measure_delta_var.set("Δ(B-A): —")
            self.measure_axis_abs_var.set("|Δ axes|: —")
            self.measure_distance_var.set("3D distance: —")
            return

        point_b_m = getEdgePointObjectPosition(self.rigid_planes, self.measurement_points[1])
        delta_m = point_b_m - point_a_m
        distance_m = float(np.linalg.norm(delta_m))
        point_b = self.metersToDisplay(point_b_m)
        delta = self.metersToDisplay(delta_m)
        distance = distance_m/self.unitToMeters()

        self.measure_b_var.set(
            f"B: {edgeSelectionName(self.measurement_points[1], self.rigid_planes)}, "
            f"({point_b[0]:.4f}, {point_b[1]:.4f}, {point_b[2]:.4f}) {self.display_unit}"
        )
        self.measure_delta_var.set(
            f"Δ(B-A): x={delta[0]:+.4f}, y={delta[1]:+.4f}, z={delta[2]:+.4f} {self.display_unit}"
        )
        self.measure_axis_abs_var.set(
            f"|Δ axes|: x={abs(delta[0]):.4f}, y={abs(delta[1]):.4f}, z={abs(delta[2]):.4f} {self.display_unit}"
        )
        self.measure_distance_var.set(f"3D distance: {distance:.4f} {self.display_unit}")

    def onMeasurementSlider(self, index: int, value: str) -> None:
        if self.updating_measure_controls or index >= len(self.measurement_points):
            return

        selection = self.measurement_points[index]
        edge_length_display = self.getMeasurementEdgeLengthM(selection)/self.unitToMeters()
        selection.t = 0.0 if edge_length_display <= 1e-12 else float(np.clip(float(value)/edge_length_display, 0.0, 1.0))
        self.measure_t_value_vars[index].set(f"s = {float(value):.3f} {self.display_unit}")
        self.redraw()

    def onMeasurementCoordinateEntered(self, point_index: int, axis_index: int) -> None:
        if point_index >= len(self.measurement_points):
            return

        entry = self.measure_coord_entries[point_index][axis_index]

        try:
            target_coordinate = float(entry.get())*self.unitToMeters()
        except ValueError:
            messagebox.showerror("Invalid coordinate", f"Enter a numeric coordinate in {self.display_unit}.")
            self.updateMeasurementControls()
            return

        selection = self.measurement_points[point_index]
        endpoint_0 = EdgePointSelection(selection.plane_index, selection.shape_index, selection.edge_index, 0.0)
        endpoint_1 = EdgePointSelection(selection.plane_index, selection.shape_index, selection.edge_index, 1.0)
        point_0 = getEdgePointObjectPosition(self.rigid_planes, endpoint_0)
        point_1 = getEdgePointObjectPosition(self.rigid_planes, endpoint_1)
        coordinate_change = point_1[axis_index] - point_0[axis_index]

        if abs(coordinate_change) <= 1e-10:
            messagebox.showerror(
                "Coordinate cannot determine position",
                f"This edge has essentially constant {('x', 'y', 'z')[axis_index]}, so that coordinate cannot determine where the point lies along the edge.",
            )
            self.updateMeasurementControls()
            return

        t = (target_coordinate - point_0[axis_index])/coordinate_change

        if t < -1e-9 or t > 1.0 + 1e-9:
            low, high = sorted((point_0[axis_index], point_1[axis_index]))
            messagebox.showerror(
                "Coordinate is outside the selected edge",
                f"{('x', 'y', 'z')[axis_index]} must be between "
                f"{low/self.unitToMeters():.6f} and {high/self.unitToMeters():.6f} {self.display_unit} for this edge.",
            )
            self.updateMeasurementControls()
            return

        selection.t = float(np.clip(t, 0.0, 1.0))
        self.redraw()

    def clearMeasurement(self) -> None:
        self.measurement_points.clear()
        self.redraw()

    def refreshPlaneList(self, select_index: int | None = None) -> None:
        self.plane_list.delete(0, tk.END)

        for index, plane in enumerate(self.rigid_planes):
            visibility = "x" if plane.visible else " "
            self.plane_list.insert(tk.END, f"[{visibility}] Plane {index}  ({len(plane.shape_markers)} shapes)")

        if select_index is not None and self.rigid_planes:
            select_index = min(select_index, len(self.rigid_planes) - 1)
            self.plane_list.selection_set(select_index)
            self.plane_list.activate(select_index)
            self.selected_plane_index = select_index
            self.loadPlaneFields()
            self.refreshShapeList()
        elif not self.rigid_planes:
            self.selected_plane_index = None
            self.selected_shape_index = None
            self.clearPlaneFields()
            self.refreshShapeList()

    def refreshShapeList(self, select_index: int | None = None) -> None:
        self.shape_list.delete(0, tk.END)
        self.selected_shape_index = None

        if self.selected_plane_index is None:
            self.clearShapeFields()
            return

        shapes = self.rigid_planes[self.selected_plane_index].shape_markers

        for index, shape in enumerate(shapes):
            visibility = "x" if shape.visible else " "
            self.shape_list.insert(tk.END, f"[{visibility}] Shape {index}: {shape.color_id.name}, {shape.num_sides} sides")

        if select_index is not None and shapes:
            select_index = min(select_index, len(shapes) - 1)
            self.shape_list.selection_set(select_index)
            self.shape_list.activate(select_index)
            self.selected_shape_index = select_index
            self.loadShapeFields()
        else:
            self.clearShapeFields()

    def onPlaneSelected(self, _event=None) -> None:
        selection = self.plane_list.curselection()
        if not selection:
            return

        self.selected_plane_index = selection[0]
        self.loadPlaneFields()
        self.refreshShapeList()

    def onShapeSelected(self, _event=None) -> None:
        selection = self.shape_list.curselection()
        if not selection:
            return

        self.selected_shape_index = selection[0]
        self.loadShapeFields()

    def onPlaneVisibilityChanged(self) -> None:
        if self.selected_plane_index is None:
            return

        self.rigid_planes[self.selected_plane_index].visible = bool(self.plane_visible_var.get())
        selected = self.selected_plane_index
        self.refreshPlaneList(selected)
        self.redraw()

    def onShapeVisibilityChanged(self) -> None:
        if self.selected_plane_index is None or self.selected_shape_index is None:
            return

        shape = self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index]
        shape.visible = bool(self.shape_visible_var.get())
        plane_index, shape_index = self.selected_plane_index, self.selected_shape_index
        self.refreshPlaneList(plane_index)
        self.refreshShapeList(shape_index)
        self.redraw()

    def clearPlaneFields(self) -> None:
        self.updating_rotation_controls = True
        self.plane_visible_var.set(True)

        for entry in self.angle_entries:
            entry.delete(0, tk.END)

        for variable in self.angle_vars:
            variable.set(0.0)

        for row in self.rotation_entries:
            for entry in row:
                entry.delete(0, tk.END)

        for entry in self.translation_entries:
            entry.delete(0, tk.END)

        self.updating_rotation_controls = False

    def loadPlaneFields(self) -> None:
        if self.selected_plane_index is None:
            return

        plane = self.rigid_planes[self.selected_plane_index]
        self.plane_visible_var.set(plane.visible)
        self.setRotationControlsFromMatrix(plane.rotation_object_from_plane)

        translation_display = self.metersToDisplay(plane.translation_object_from_plane_m)
        for index, entry in enumerate(self.translation_entries):
            entry.delete(0, tk.END)
            entry.insert(0, f"{translation_display[index]:.8g}")

    def setRotationControlsFromMatrix(self, rotation: np.ndarray) -> None:
        self.updating_rotation_controls = True
        angles = anglesDegFromRotationMatrix(rotation)

        for index, angle in enumerate(angles):
            self.angle_vars[index].set(float(angle))
            self.angle_entries[index].delete(0, tk.END)
            self.angle_entries[index].insert(0, f"{angle:.3f}")

        for row in range(3):
            for col in range(3):
                entry = self.rotation_entries[row][col]
                entry.delete(0, tk.END)
                entry.insert(0, f"{rotation[row, col]:.8g}")

        self.updating_rotation_controls = False

    def applyAnglesToSelectedPlane(self) -> None:
        if self.selected_plane_index is None or self.updating_rotation_controls:
            return

        try:
            angles = [float(entry.get()) for entry in self.angle_entries]
        except ValueError:
            return

        rotation = rotationMatrixFromAnglesDeg(*angles)
        self.rigid_planes[self.selected_plane_index].rotation_object_from_plane = rotation
        self.setRotationControlsFromMatrix(rotation)
        self.redraw()

    def onAngleSliderChanged(self, axis_index: int) -> None:
        if self.selected_plane_index is None or self.updating_rotation_controls:
            return

        self.updating_rotation_controls = True
        angle = self.angle_vars[axis_index].get()
        self.angle_entries[axis_index].delete(0, tk.END)
        self.angle_entries[axis_index].insert(0, f"{angle:.3f}")
        self.updating_rotation_controls = False
        self.applyAnglesToSelectedPlane()

    def onAngleEntryChanged(self, axis_index: int) -> None:
        if self.selected_plane_index is None or self.updating_rotation_controls:
            return

        try:
            angle = float(self.angle_entries[axis_index].get())
        except ValueError:
            return

        angle = max(-180.0, min(180.0, angle))
        self.angle_vars[axis_index].set(angle)
        self.applyAnglesToSelectedPlane()

    def clearShapeFields(self) -> None:
        self.shape_visible_var.set(True)
        self.points_text.delete("1.0", tk.END)

    def loadShapeFields(self) -> None:
        if self.selected_plane_index is None or self.selected_shape_index is None:
            return

        shape = self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index]
        self.shape_visible_var.set(shape.visible)
        self.color_var.set(shape.color_id.name)
        points_display = self.metersToDisplay(shape.object_vertices_m)
        self.points_text.delete("1.0", tk.END)
        self.points_text.insert("1.0", "\n".join(f"{x:.8g} {y:.8g}" for x, y in points_display))

    def addPlane(self) -> None:
        self.rigid_planes.append(EditableRigidPlane(np.eye(3), np.zeros(3)))
        self.refreshPlaneList(len(self.rigid_planes) - 1)
        self.redraw()

    def deletePlane(self) -> None:
        if self.selected_plane_index is None:
            return

        index = self.selected_plane_index
        self.measurement_points.clear()
        del self.rigid_planes[index]
        self.refreshPlaneList(min(index, len(self.rigid_planes) - 1) if self.rigid_planes else None)
        self.redraw()

    def applyPlane(self) -> None:
        if self.selected_plane_index is None:
            return

        try:
            rotation = parseRotation(self.rotation_entries)
            translation = parseTranslation(self.translation_entries, self.unitToMeters())
        except ValueError as error:
            messagebox.showerror("Invalid plane transform", str(error))
            return

        plane = self.rigid_planes[self.selected_plane_index]
        plane.rotation_object_from_plane = rotation
        plane.translation_object_from_plane_m = translation
        self.setRotationControlsFromMatrix(rotation)
        self.redraw()

    def addShape(self) -> None:
        if self.selected_plane_index is None:
            messagebox.showinfo("Select a plane", "Add or select a rigid plane first.")
            return

        default_points = np.array([[0.02, 0.02], [-0.02, 0.02], [-0.02, -0.02], [0.02, -0.02]])
        shape = EditableShape(ColorId[self.color_var.get()], default_points)
        shapes = self.rigid_planes[self.selected_plane_index].shape_markers
        shapes.append(shape)
        self.refreshPlaneList(self.selected_plane_index)
        self.refreshShapeList(len(shapes) - 1)
        self.redraw()

    def deleteShape(self) -> None:
        if self.selected_plane_index is None or self.selected_shape_index is None:
            return

        plane_index, shape_index = self.selected_plane_index, self.selected_shape_index
        self.measurement_points.clear()
        shapes = self.rigid_planes[plane_index].shape_markers
        del shapes[shape_index]
        self.refreshPlaneList(plane_index)
        self.refreshShapeList(min(shape_index, len(shapes) - 1) if shapes else None)
        self.redraw()

    def applyShape(self) -> None:
        if self.selected_plane_index is None or self.selected_shape_index is None:
            return

        try:
            color_id = ColorId[self.color_var.get()]
            points = parsePoints(self.points_text.get("1.0", tk.END), self.unitToMeters())
        except (ValueError, KeyError) as error:
            messagebox.showerror("Invalid shape", str(error))
            return

        old_shape = self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index]
        self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index] = EditableShape(
            color_id, points, old_shape.minimum_contour_area_px, old_shape.visible,
        )
        self.refreshPlaneList(self.selected_plane_index)
        self.refreshShapeList(self.selected_shape_index)
        self.redraw()

    def showCode(self) -> None:
        code = objectVisionSpecCode(
            self.object_vision_spec_id.name if self.object_vision_spec_id is not None else self.model_name.upper(),
            self.currentObjectVisionSpec(),
        )
        window = tk.Toplevel(self)
        window.title("Generated ObjectVisionSpec code")
        window.geometry("850x650")

        text = tk.Text(window, wrap="none")
        text.pack(fill="both", expand=True)
        text.insert("1.0", code)

        button_frame = ttk.Frame(window, padding=6)
        button_frame.pack(fill="x")

        def copyCode():
            self.clipboard_clear()
            self.clipboard_append(code)
            self.update()
            messagebox.showinfo("Copied", "Model code copied to clipboard.", parent=window)

        ttk.Button(button_frame, text="Copy to clipboard", command=copyCode).pack(side="right")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI editor for ObjectVisionSpec rigid-plane marker models.")
    parser.add_argument(
        "--spec", default=DEFAULT_OBJECT_VISION_SPEC_ID.name,
        choices=[spec_id.name for spec_id in ObjectVisionSpecId],
        help="registered ObjectVisionSpecId to open",
    )
    args = parser.parse_args()
    ModelEditor(ObjectVisionSpecId[args.spec]).mainloop()