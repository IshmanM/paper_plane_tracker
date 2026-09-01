import math
from pathlib import Path

import cv2

from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectVisionSpecId


OUTPUT_DIR = Path("images/aruco_markers")
MARKER_IMAGE_SIZE_PX = 1000

# PDF layout. A 0.25-inch margin keeps content inside the printable area of
# typical non-borderless Letter printers. Increase it if your printer requires more.
PRINTABLE_MARGIN_MM = 6.35
MARKER_GAP_MM = 3.0
PDF_HEADER_HEIGHT_MM = 10.0


def _choose_object_vision_spec_id():
    spec_ids = list(ObjectVisionSpecId)

    print("Available object vision spec IDs:")
    for index, spec_id in enumerate(spec_ids, start=1):
        print(f"  {index}: {spec_id.name}")

    while True:
        choice = input("Choose an object vision spec ID by number or name: ").strip()

        selected_spec_id = None
        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(spec_ids):
                selected_spec_id = spec_ids[selected_index]
        else:
            choice_upper = choice.upper()
            selected_spec_id = next(
                (spec_id for spec_id in spec_ids if spec_id.name.upper() == choice_upper),
                None,
            )

        if selected_spec_id is None:
            print("Invalid selection. Enter one of the displayed numbers or names.")
            continue

        object_vision_spec = OBJECT_VISION_SPECS.get(selected_spec_id)
        if object_vision_spec is None:
            print(f"{selected_spec_id.name} is missing from OBJECT_VISION_SPECS.")
            continue

        aruco_spec = getattr(object_vision_spec, "aruco_marker", None)
        if aruco_spec is None:
            print(
                f"{selected_spec_id.name} does not have ArUco marker information. "
                "Choose another spec."
            )
            continue

        return selected_spec_id, aruco_spec


def _choose_print_copies(page_capacity):
    while True:
        choice = input(
            f"How many markers should be printed on the page? "
            f"[1-{page_capacity}, Enter={page_capacity}]: "
        ).strip()

        if not choice:
            return page_capacity

        try:
            copies = int(choice)
        except ValueError:
            copies = 0

        if 1 <= copies <= page_capacity:
            return copies

        print(f"Invalid quantity. Enter a number from 1 to {page_capacity}.")


def _get_pdf_grid_capacity(page_width, page_height, marker_length, margin, gap, header_height):
    usable_width = page_width - 2.0 * margin
    usable_height = page_height - 2.0 * margin - header_height

    if marker_length > usable_width or marker_length > usable_height:
        raise ValueError("The configured marker is too large to fit on the PDF page.")

    columns = max(1, math.floor((usable_width + gap) / (marker_length + gap)))
    rows = max(1, math.floor((usable_height + gap) / (marker_length + gap)))
    return columns, rows


def main():
    object_vision_spec_id, aruco_spec = _choose_object_vision_spec_id()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_spec.dictionary_name))
    marker_image = cv2.aruco.generateImageMarker(dictionary, aruco_spec.marker_id, MARKER_IMAGE_SIZE_PX)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename_base = (
        f"{object_vision_spec_id.name.lower()}_"
        f"{aruco_spec.dictionary_name.lower()}_id{aruco_spec.marker_id}"
    )
    png_path = OUTPUT_DIR/f"{filename_base}.png"
    pdf_path = OUTPUT_DIR/f"{filename_base}_print.pdf"

    if not cv2.imwrite(str(png_path), marker_image):
        raise RuntimeError(f"Failed to save {png_path}")

    marker_length_mm = aruco_spec.marker_length_m*1000.0

    print(f"Saved PNG: {png_path}")
    print(f"Dictionary: {aruco_spec.dictionary_name}")
    print(f"Marker ID: {aruco_spec.marker_id}")
    print(f"Marker side length: {marker_length_mm:.2f} mm")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas

        page_width, page_height = letter
        marker_length = marker_length_mm * mm

        margin = PRINTABLE_MARGIN_MM * mm
        gap = MARKER_GAP_MM * mm
        header_height = PDF_HEADER_HEIGHT_MM * mm
        max_columns, max_rows = _get_pdf_grid_capacity(
            page_width,
            page_height,
            marker_length,
            margin,
            gap,
            header_height,
        )
        page_capacity = max_columns * max_rows
        print(
            f"A maximum of {page_capacity} marker(s) fit inside the "
            f"{PRINTABLE_MARGIN_MM:.2f} mm printable margins."
        )
        copies = _choose_print_copies(page_capacity)

        columns_used = min(max_columns, copies)
        rows_used = math.ceil(copies / columns_used)
        grid_height = rows_used * marker_length + (rows_used - 1) * gap
        content_bottom = margin
        content_top = page_height - margin - header_height
        grid_bottom = content_bottom + (
            content_top - content_bottom - grid_height
        ) / 2.0

        pdf = canvas.Canvas(str(pdf_path), pagesize=letter)

        remaining = copies
        for row in range(rows_used):
            markers_in_row = min(columns_used, remaining)
            row_width = markers_in_row * marker_length + (markers_in_row - 1) * gap
            row_left = (page_width - row_width) / 2.0
            y = grid_bottom + (rows_used - row - 1) * (marker_length + gap)

            for column in range(markers_in_row):
                x = row_left + column * (marker_length + gap)
                pdf.drawImage(
                    str(png_path),
                    x,
                    y,
                    width=marker_length,
                    height=marker_length,
                    preserveAspectRatio=True,
                    mask="auto",
                )

            remaining -= markers_in_row

        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(
            page_width / 2.0,
            page_height - margin - 3 * mm,
            f"ArUco {aruco_spec.dictionary_name}, ID {aruco_spec.marker_id} - "
            f"outer square: {marker_length_mm:.2f} mm x {marker_length_mm:.2f} mm",
        )

        pdf.save()

        print(f"Saved print-ready PDF: {pdf_path}")
        print(f"PDF marker copies: {copies} ({columns_used} column(s) x {rows_used} row(s))")
        print("Print the PDF using 'Actual size' or 100% scale.")
        print("Do NOT use 'Fit to page' or 'Shrink to fit'.")

    except ImportError:
        print()
        print("PDF was not generated because reportlab is not installed.")
        print("The PNG is still valid.")
        print("To enable PDF generation:")
        print("    pip install reportlab")


if __name__ == "__main__":
    main()
