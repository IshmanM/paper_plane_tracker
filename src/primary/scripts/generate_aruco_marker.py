from pathlib import Path

import cv2

from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectVisionSpecId


OBJECT_VISION_SPEC_ID = ObjectVisionSpecId.ARUCO_MARKER_1
OUTPUT_DIR = Path("images/aruco_markers")
MARKER_IMAGE_SIZE_PX = 1000


def main():
    object_vision_spec = OBJECT_VISION_SPECS[OBJECT_VISION_SPEC_ID]
    aruco_spec = object_vision_spec.aruco_marker

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_spec.dictionary_name))
    marker_image = cv2.aruco.generateImageMarker(dictionary, aruco_spec.marker_id, MARKER_IMAGE_SIZE_PX)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename_base = f"{OBJECT_VISION_SPEC_ID.name.lower()}_{aruco_spec.dictionary_name.lower()}_id{aruco_spec.marker_id}"
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
        marker_length = marker_length_mm*mm

        x = (page_width - marker_length)/2.0
        y = (page_height - marker_length)/2.0

        pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
        pdf.drawImage(
            str(png_path),
            x, y,
            width=marker_length,
            height=marker_length,
            preserveAspectRatio=True,
            mask="auto",
        )

        pdf.setFont("Helvetica", 10)
        pdf.drawCentredString(
            page_width/2.0,
            y - 10*mm,
            f"ArUco {aruco_spec.dictionary_name}, ID {aruco_spec.marker_id}"
        )
        pdf.drawCentredString(
            page_width/2.0,
            y - 15*mm,
            f"Outer marker square should measure {marker_length_mm:.2f} mm x {marker_length_mm:.2f} mm"
        )

        pdf.save()

        print(f"Saved print-ready PDF: {pdf_path}")
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