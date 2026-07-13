from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp",
}


class ImageSorter:

    def __init__(self, source_dir: str):
        self.source_path = Path(source_dir)

    def sort(self) -> int:
        if not self.source_path.exists():
            logger.warning("Source path %s does not exist.", self.source_path)
            return 0
        files = [
            f
            for f in self.source_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        moved = sum(self._process_image(f) for f in files)
        logger.info("Sorted %d images in %s", moved, self.source_path)
        return moved

    def _get_dimensions(self, file_path: Path) -> tuple[int, int] | None:
        try:
            with Image.open(file_path) as img:
                return img.size
        except OSError:
            return None

    def _move_to_target(self, file_path: Path, dimensions: tuple[int, int]) -> bool:
        width, height = dimensions
        target_dir = self.source_path / f"{width}x{height}"
        target_dir.mkdir(exist_ok=True)
        destination = target_dir / file_path.name
        if destination.exists():
            return False
        shutil.move(str(file_path), str(destination))
        return True

    def _process_image(self, file_path: Path) -> bool:
        dims = self._get_dimensions(file_path)
        if dims is None:
            return False
        return self._move_to_target(file_path, dims)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    ImageSorter(target_path).sort()


if __name__ == "__main__":
    main()