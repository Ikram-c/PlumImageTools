from __future__ import annotations

import logging
import re
from typing import Union

import exifread
from bs4 import BeautifulSoup
from iptcinfo3 import IPTCInfo

logger = logging.getLogger(__name__)


class MetadataExtractor:

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img = None

    def run_metadata_extractor(self) -> dict:
        with open(self.image_path, "rb") as img_file:
            self.img = img_file
            exif_dict = self.read_exif_metadata()
            xmp_dict = self.xmp_metadata_cleaner()
            iptc_dict = self.read_iptc_metadata()

            exif_dict = {str(k): str(v) for k, v in exif_dict.items()}

            if isinstance(xmp_dict, dict):
                metadata_dict = {**exif_dict, **xmp_dict, **iptc_dict}
            else:
                metadata_dict = {**exif_dict, **iptc_dict, "XMP_error": xmp_dict}

            return {str(k): str(v) for k, v in metadata_dict.items()}

    def read_exif_metadata(self) -> dict:
        if hasattr(self.img, "seek"):
            self.img.seek(0)
        return exifread.process_file(self.img, details=False)

    def read_iptc_metadata(self) -> dict:
        if hasattr(self.img, "seek"):
            self.img.seek(0)
        iptc_info = IPTCInfo(self.img)
        return {
            k: v
            for k, v in iptc_info._data.items()
            if v not in (None, b"", "")
        }

    def read_xmp_metadata(self) -> Union[dict, str]:
        if hasattr(self.img, "seek"):
            self.img.seek(0)
        data = self.img.read() if hasattr(self.img, "read") else self.img

        xmp_start = data.find(b"<x:xmpmeta")
        xmp_end = data.find(b"</x:xmpmeta>")
        if -1 in (xmp_start, xmp_end):
            return "No XMP metadata found in the image."

        xmp_end += len(b"</x:xmpmeta>")
        xmp_string = data[xmp_start:xmp_end].decode("utf-8", errors="ignore")
        xmp_as_xml = BeautifulSoup(xmp_string, "xml")
        rdf_description = xmp_as_xml.find("rdf:Description")
        if not rdf_description:
            return "No rdf:Description found in the XMP metadata."
        return dict(rdf_description.attrs)

    @staticmethod
    def _convert_value(value: Union[str, int, float]) -> Union[float, str]:
        if isinstance(value, (int, float)):
            return float(value)
        is_numeric_string = isinstance(value, str) and re.match(
            r"^[+-]?\d*\.?\d+$", value.strip()
        )
        if is_numeric_string:
            return float(value)
        return str(value)

    def xmp_metadata_cleaner(self) -> Union[dict, str]:
        xmp_data = self.read_xmp_metadata()
        if isinstance(xmp_data, str):
            return xmp_data
        return {
            key.split(":")[-1]: self._convert_value(value)
            for key, value in xmp_data.items()
        }