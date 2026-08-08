import argparse
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType
from src.primary.color import COLOR_SPECS, ColorId


DEFAULT_OBJECT_TYPE = ObjectType.PAPER_PLANE_SHAPES
PLANE_MARGIN_M = 0.015
EMPTY_PLANE_HALF_SIZE_M = 0.03
OBJECT_AXIS_LENGTH_M = 0.08
PLANE_AXIS_LENGTH_M = 0.025


class EditableShape:
    def __init__(self, color_id: ColorId, vertices_xy_m: np.ndarray, minimum_contour_area_px: float | None = None):
        self.color_id = color_id
        self.object_vertices_m = np.asarray(vertices_xy_m, dtype=np.float64)
        self.minimum_contour_area_px = minimum_contour_area_px

    @property
    def num_sides(self) -> int:
        return len(self.object_vertices_m)


class EditableRigidPlane:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, shape_markers: list[EditableShape] | None = None):
        self.rotation_object_from_plane = np.asarray(rotation, dtype=np.float64)
        self.translation_object_from_plane_m = np.asarray(translation, dtype=np.float64)
        self.shape_markers = shape_markers if shape_markers is not None else []


def copyModelFromSpec(object_type: ObjectType) -> list[EditableRigidPlane]:
    rigid_planes = []

    for rigid_plane in OBJECT_VISION_SPECS[object_type].rigid_planes:
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


def setAxesEqual(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs)/2.0
    radius = max(np.max(maxs - mins)/2.0, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def drawModel(
    ax, rigid_planes: list[EditableRigidPlane], object_type: ObjectType,
    view_elev: float = 25.0, view_azim: float = -55.0, view_roll: float = 0.0,
    vertex_pick_map: dict | None = None, selected_vertex_keys: list[tuple[int, int, int]] | None = None,
) -> None:
    ax.clear()
    all_points = [np.zeros(3)]
    selected_vertex_keys = selected_vertex_keys if selected_vertex_keys is not None else []
    selected_labels = {key: ("A" if index == 0 else "B") for index, key in enumerate(selected_vertex_keys[:2])}

    if vertex_pick_map is not None:
        vertex_pick_map.clear()

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
        all_points.append(translation)

        # Plane-local +x, +y, and normal transformed into the object frame.
        local_axes_object = rotation@np.eye(3)

        for axis_index, (label, color) in enumerate(zip(
            (f"P{plane_index} +x", f"P{plane_index} +y", f"P{plane_index} normal"),
            ("tab:red", "tab:green", "tab:blue"),
        )):
            vector = local_axes_object[:, axis_index]*PLANE_AXIS_LENGTH_M
            ax.quiver(*translation, *vector, color=color, linewidth=1.4, arrow_length_ratio=0.15)
            ax.text(*(translation + vector*1.1), label, color=color, fontsize=8)

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
        ax.add_collection3d(Poly3DCollection([patch_object], alpha=0.10, edgecolor="0.45", linewidth=1))

        # Marker points are plane-local (x, y), then lifted to z=0, rotated, and translated.
        for shape_index, shape in enumerate(rigid_plane.shape_markers):
            vertices_object = transformPlanePoints(shape.object_vertices_m, rotation, translation)
            all_points.extend(vertices_object)
            closed_vertices = np.vstack((vertices_object, vertices_object[0]))

            b, g, r = COLOR_SPECS[shape.color_id].draw_bgr
            marker_color = (r/255.0, g/255.0, b/255.0)
            ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], closed_vertices[:, 2], color=marker_color, linewidth=3)
            vertex_artist = ax.scatter(
                vertices_object[:, 0], vertices_object[:, 1], vertices_object[:, 2],
                color=[marker_color], s=55, picker=8,
            )

            if vertex_pick_map is not None:
                vertex_pick_map[vertex_artist] = [
                    ((plane_index, shape_index, vertex_index), vertex.copy())
                    for vertex_index, vertex in enumerate(vertices_object)
                ]

            center = vertices_object.mean(axis=0)
            ax.text(*center, f"P{plane_index} S{shape_index} {shape.color_id.name}", color=marker_color, fontsize=9)

            for vertex_index, vertex in enumerate(vertices_object):
                vertex_key = (plane_index, shape_index, vertex_index)
                ax.text(*vertex, f" {vertex_index}", color=marker_color, fontsize=8)

                if vertex_key in selected_labels:
                    ax.scatter(
                        [vertex[0]], [vertex[1]], [vertex[2]], s=130,
                        facecolors="none", edgecolors="black", linewidths=2,
                    )
                    ax.text(*vertex, f"  {selected_labels[vertex_key]}", color="black", fontsize=10, fontweight="bold")

    setAxesEqual(ax, np.asarray(all_points))
    ax.set_xlabel("Object x [m] — forward")
    ax.set_ylabel("Object y [m] — down")
    ax.set_zlabel("Object z [m] — left")
    ax.set_title(f"{object_type.name}\np_object = R @ [x, y, 0] + t")

    try:
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
    except TypeError:  # Older Matplotlib versions do not expose roll.
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


def parseTranslation(entries: list[tk.Entry]) -> np.ndarray:
    return np.array([float(entry.get()) for entry in entries], dtype=np.float64)


def parsePoints(text: str) -> np.ndarray:
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

    return np.asarray(points, dtype=np.float64)


def modelCode(rigid_planes: list[EditableRigidPlane]) -> str:
    lines = ["rigid_planes=["]

    for plane in rigid_planes:
        lines += [
            "    RigidPlaneSpec(",
            "        rotation_object_from_plane=np.array([",
            *[f"            {row.tolist()}," for row in plane.rotation_object_from_plane],
            "        ], dtype=np.float64),",
            f"        translation_object_from_plane_m=np.array({plane.translation_object_from_plane_m.tolist()}, dtype=np.float64),",
            "        shape_markers=[",
        ]

        for shape in plane.shape_markers:
            lines += [
                "            ShapeMarkerSpec(",
                f"                color_id=ColorId.{shape.color_id.name},",
                f"                object_vertices_m={shape.object_vertices_m.tolist()},",
                f"                num_sides={shape.num_sides},",
            ]
            if shape.minimum_contour_area_px is not None:
                lines.append(f"                minimum_contour_area_px={shape.minimum_contour_area_px},")
            lines.append("            ),")

        lines += ["        ],", "    ),"]

    lines.append("]")
    return "\n".join(lines)


class ModelEditor(tk.Tk):
    def __init__(self, object_type: ObjectType):
        super().__init__()
        self.title("Object Vision Model Editor")
        self.geometry("1450x850")
        self.minsize(1100, 700)

        self.object_type = object_type
        self.rigid_planes = copyModelFromSpec(object_type)
        self.selected_plane_index: int | None = None
        self.selected_shape_index: int | None = None

        # Clickable marker-vertex measurement state.
        self.vertex_pick_map: dict = {}
        self.measure_vertex_keys: list[tuple[int, int, int]] = []

        # Keep the 3D camera independent of model redraws. ax.clear() resets the
        # Matplotlib 3D view, so redraw() always restores these saved angles.
        self.view_elev = 25.0
        self.view_azim = -55.0
        self.view_roll = 0.0

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
        self.refreshPlaneList()
        self.redraw()

    def buildControls(self) -> None:
        ttk.Label(self.controls, text="Rigid planes").grid(row=0, column=0, columnspan=3, sticky="w")
        self.plane_list = tk.Listbox(self.controls, height=6, exportselection=False)
        self.plane_list.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(3, 5))
        self.plane_list.bind("<<ListboxSelect>>", self.onPlaneSelected)

        ttk.Button(self.controls, text="Add plane", command=self.addPlane).grid(row=2, column=0, sticky="ew")
        ttk.Button(self.controls, text="Delete plane", command=self.deletePlane).grid(row=2, column=1, sticky="ew")
        ttk.Button(self.controls, text="Apply plane", command=self.applyPlane).grid(row=2, column=2, sticky="ew")

        ttk.Label(self.controls, text="Rotation angles [deg]").grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Label(self.controls, text="Convention: R = Rz(z) @ Ry(y) @ Rx(x)").grid(row=4, column=0, columnspan=3, sticky="w")

        self.angle_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.angle_entries, self.angle_scales = [], []
        self.updating_rotation_controls = False

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            row = 5 + axis_index
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

        ttk.Label(self.controls, text="Rotation matrix R").grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 2))
        self.rotation_entries = []

        for row in range(3):
            entry_row = []
            for col in range(3):
                entry = ttk.Entry(self.controls, width=10)
                entry.grid(row=9 + row, column=col, padx=2, pady=2, sticky="ew")
                entry_row.append(entry)
            self.rotation_entries.append(entry_row)

        ttk.Label(self.controls, text="Translation t [m]").grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.translation_entries = []

        for col, axis_name in enumerate(("x", "y", "z")):
            frame = ttk.Frame(self.controls)
            frame.grid(row=13, column=col, padx=2, sticky="ew")
            ttk.Label(frame, text=axis_name).pack(side="left")
            entry = ttk.Entry(frame, width=9)
            entry.pack(side="left", fill="x", expand=True)
            self.translation_entries.append(entry)

        ttk.Separator(self.controls).grid(row=14, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(self.controls, text="Shapes on selected plane").grid(row=15, column=0, columnspan=3, sticky="w")
        self.shape_list = tk.Listbox(self.controls, height=6, exportselection=False)
        self.shape_list.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(3, 5))
        self.shape_list.bind("<<ListboxSelect>>", self.onShapeSelected)

        ttk.Button(self.controls, text="Add shape", command=self.addShape).grid(row=17, column=0, sticky="ew")
        ttk.Button(self.controls, text="Delete shape", command=self.deleteShape).grid(row=17, column=1, sticky="ew")
        ttk.Button(self.controls, text="Apply shape", command=self.applyShape).grid(row=17, column=2, sticky="ew")

        ttk.Label(self.controls, text="Color").grid(row=18, column=0, sticky="w", pady=(8, 2))
        self.color_var = tk.StringVar(value=next(iter(ColorId.__members__)))
        self.color_combo = ttk.Combobox(self.controls, textvariable=self.color_var, values=list(ColorId.__members__), state="readonly")
        self.color_combo.grid(row=18, column=1, columnspan=2, sticky="ew", pady=(8, 2))

        ttk.Label(self.controls, text="Plane-local points [m] — one x y pair per line").grid(row=19, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.points_text = tk.Text(self.controls, width=36, height=7)
        self.points_text.grid(row=20, column=0, columnspan=3, sticky="ew")

        ttk.Button(self.controls, text="Show / copy spec code", command=self.showCode).grid(row=21, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Button(self.controls, text="Reset view", command=self.resetView).grid(row=22, column=0, columnspan=3, sticky="ew")

        for col in range(3):
            self.controls.columnconfigure(col, weight=1)

    def buildPlot(self) -> None:
        self.figure = plt.Figure(figsize=(9, 8))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.setPlotView()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_release_event", self.savePlotView)
        self.canvas.mpl_connect("pick_event", self.onVertexPicked)

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        measurement_frame = ttk.LabelFrame(self.plot_frame, text="Vertex measurement", padding=6)
        measurement_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        measurement_frame.columnconfigure(0, weight=1)

        self.measure_a_var = tk.StringVar(value="A: click a marker vertex")
        self.measure_b_var = tk.StringVar(value="B: click a second marker vertex")
        self.measure_delta_var = tk.StringVar(value="Δ(B-A): —")
        self.measure_axis_abs_var = tk.StringVar(value="|Δ axes|: —")
        self.measure_distance_var = tk.StringVar(value="3D distance: —")

        ttk.Label(measurement_frame, textvariable=self.measure_a_var).grid(row=0, column=0, sticky="w")
        ttk.Label(measurement_frame, textvariable=self.measure_b_var).grid(row=1, column=0, sticky="w")
        ttk.Label(measurement_frame, textvariable=self.measure_delta_var).grid(row=2, column=0, sticky="w")
        ttk.Label(measurement_frame, textvariable=self.measure_axis_abs_var).grid(row=3, column=0, sticky="w")
        ttk.Label(measurement_frame, textvariable=self.measure_distance_var).grid(row=4, column=0, sticky="w")
        ttk.Button(measurement_frame, text="Clear measurement", command=self.clearVertexMeasurement).grid(
            row=0, column=1, rowspan=5, padx=(12, 0), sticky="ns",
        )

    def savePlotView(self, _event=None) -> None:
        self.view_elev = float(self.ax.elev)
        self.view_azim = float(self.ax.azim)
        self.view_roll = float(getattr(self.ax, "roll", 0.0))

    def setPlotView(self) -> None:
        try:
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim, roll=self.view_roll)
        except TypeError:
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim)

    def redraw(self) -> None:
        # Capture the live camera before clearing/redrawing the model. This also
        # covers a redraw that happens before the mouse-release callback fires.
        self.savePlotView()
        self.updateVertexMeasurement()
        drawModel(
            self.ax, self.rigid_planes, self.object_type,
            self.view_elev, self.view_azim, self.view_roll,
            self.vertex_pick_map, self.measure_vertex_keys,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def getVertexObjectPosition(self, vertex_key: tuple[int, int, int]) -> np.ndarray | None:
        try:
            plane_index, shape_index, vertex_index = vertex_key
            plane = self.rigid_planes[plane_index]
            shape = plane.shape_markers[shape_index]
            vertex_xy = shape.object_vertices_m[vertex_index]
        except IndexError:
            return None

        return transformPlanePoints(
            np.asarray([vertex_xy], dtype=np.float64),
            plane.rotation_object_from_plane, plane.translation_object_from_plane_m,
        )[0]

    def vertexName(self, vertex_key: tuple[int, int, int]) -> str:
        plane_index, shape_index, vertex_index = vertex_key
        return f"P{plane_index} S{shape_index} V{vertex_index}"

    def onVertexPicked(self, event) -> None:
        vertex_options = self.vertex_pick_map.get(event.artist)

        if not vertex_options or len(event.ind) == 0:
            return

        picked_index = int(event.ind[0])
        if picked_index >= len(vertex_options):
            return

        vertex_key, _ = vertex_options[picked_index]

        # First two clicks form A/B. A third click starts a new measurement.
        if len(self.measure_vertex_keys) >= 2:
            self.measure_vertex_keys = [vertex_key]
        else:
            self.measure_vertex_keys.append(vertex_key)

        self.redraw()

    def clearVertexMeasurement(self) -> None:
        self.measure_vertex_keys.clear()
        self.updateVertexMeasurement()
        self.redraw()

    def updateVertexMeasurement(self) -> None:
        valid_keys = [
            key for key in self.measure_vertex_keys
            if self.getVertexObjectPosition(key) is not None
        ]
        self.measure_vertex_keys = valid_keys[:2]

        if not hasattr(self, "measure_a_var"):
            return

        if not self.measure_vertex_keys:
            self.measure_a_var.set("A: click a marker vertex")
            self.measure_b_var.set("B: click a second marker vertex")
            self.measure_delta_var.set("Δ(B-A): —")
            self.measure_axis_abs_var.set("|Δ axes|: —")
            self.measure_distance_var.set("3D distance: —")
            return

        point_a = self.getVertexObjectPosition(self.measure_vertex_keys[0])
        self.measure_a_var.set(
            f"A: {self.vertexName(self.measure_vertex_keys[0])} = "
            f"({point_a[0]:.5f}, {point_a[1]:.5f}, {point_a[2]:.5f}) m"
        )

        if len(self.measure_vertex_keys) < 2:
            self.measure_b_var.set("B: click a second marker vertex")
            self.measure_delta_var.set("Δ(B-A): —")
            self.measure_axis_abs_var.set("|Δ axes|: —")
            self.measure_distance_var.set("3D distance: —")
            return

        point_b = self.getVertexObjectPosition(self.measure_vertex_keys[1])
        delta = point_b - point_a
        distance = float(np.linalg.norm(delta))

        self.measure_b_var.set(
            f"B: {self.vertexName(self.measure_vertex_keys[1])} = "
            f"({point_b[0]:.5f}, {point_b[1]:.5f}, {point_b[2]:.5f}) m"
        )
        self.measure_delta_var.set(
            f"Δ(B-A): x={delta[0]:+.5f}, y={delta[1]:+.5f}, z={delta[2]:+.5f} m"
        )
        self.measure_axis_abs_var.set(
            f"|Δ axes|: x={abs(delta[0]):.5f}, y={abs(delta[1]):.5f}, z={abs(delta[2]):.5f} m"
        )
        self.measure_distance_var.set(
            f"3D distance: {distance:.5f} m  ({100.0*distance:.2f} cm)"
        )

    def refreshPlaneList(self, select_index: int | None = None) -> None:
        self.plane_list.delete(0, tk.END)

        for index, plane in enumerate(self.rigid_planes):
            self.plane_list.insert(tk.END, f"Plane {index}  ({len(plane.shape_markers)} shapes)")

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
            self.shape_list.insert(tk.END, f"Shape {index}: {shape.color_id.name}, {shape.num_sides} sides")

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

    def clearPlaneFields(self) -> None:
        self.updating_rotation_controls = True

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
        self.setRotationControlsFromMatrix(plane.rotation_object_from_plane)

        for index, entry in enumerate(self.translation_entries):
            entry.delete(0, tk.END)
            entry.insert(0, f"{plane.translation_object_from_plane_m[index]:.8g}")

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
        self.points_text.delete("1.0", tk.END)

    def loadShapeFields(self) -> None:
        if self.selected_plane_index is None or self.selected_shape_index is None:
            return

        shape = self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index]
        self.color_var.set(shape.color_id.name)
        self.points_text.delete("1.0", tk.END)
        self.points_text.insert("1.0", "\n".join(f"{x:.8g} {y:.8g}" for x, y in shape.object_vertices_m))

    def addPlane(self) -> None:
        self.rigid_planes.append(EditableRigidPlane(np.eye(3), np.zeros(3)))
        self.refreshPlaneList(len(self.rigid_planes) - 1)
        self.redraw()

    def deletePlane(self) -> None:
        if self.selected_plane_index is None:
            return

        index = self.selected_plane_index
        self.measure_vertex_keys.clear()
        del self.rigid_planes[index]
        self.refreshPlaneList(min(index, len(self.rigid_planes) - 1) if self.rigid_planes else None)
        self.redraw()

    def applyPlane(self) -> None:
        if self.selected_plane_index is None:
            return

        try:
            rotation = parseRotation(self.rotation_entries)
            translation = parseTranslation(self.translation_entries)
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
        self.measure_vertex_keys.clear()
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
            points = parsePoints(self.points_text.get("1.0", tk.END))
        except (ValueError, KeyError) as error:
            messagebox.showerror("Invalid shape", str(error))
            return

        old_shape = self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index]
        self.rigid_planes[self.selected_plane_index].shape_markers[self.selected_shape_index] = EditableShape(
            color_id, points, old_shape.minimum_contour_area_px,
        )
        self.refreshPlaneList(self.selected_plane_index)
        self.refreshShapeList(self.selected_shape_index)
        self.redraw()

    def showCode(self) -> None:
        code = modelCode(self.rigid_planes)
        window = tk.Toplevel(self)
        window.title("Generated model code")
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

    def resetView(self) -> None:
        self.view_elev = 25.0
        self.view_azim = -55.0
        self.view_roll = 0.0
        self.setPlotView()
        self.canvas.draw_idle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI editor for ObjectVisionSpec rigid-plane marker models.")
    parser.add_argument("--object", default=DEFAULT_OBJECT_TYPE.name, choices=[object_type.name for object_type in ObjectType])
    args = parser.parse_args()
    ModelEditor(ObjectType[args.object]).mainloop()