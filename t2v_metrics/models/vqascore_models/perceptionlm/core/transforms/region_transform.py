import re
from typing import Any, Callable, Dict, List, Tuple


def get_region_transform(
    region_type: str = "bbox",
    region_format: str = "xyxy",
    coordinate_format: str = "000",
    coord_decimal: int = 3,
) -> Tuple[Callable, int]:

    transforms = RegionTransform(
        region_type=region_type,
        region_format=region_format,
        coordinate_format=coordinate_format,
        coord_decimal=coord_decimal,
    )

    return transforms


class RegionTransform(object):
    def __init__(
        self,
        region_type: str = "bbox",
        region_format: str = "xyxy",
        coordinate_format: str = "000",
        coord_decimal: int = 3,
    ):
        assert region_type in ["bbox", "mask"]
        assert region_format in ["xyxy", "xywh", "polygon"]
        assert coordinate_format in ["000", "standard"]

        self.region_type = region_type
        self.region_format = region_format
        self.coordinate_format = coordinate_format
        self.coord_decimal = coord_decimal

    def clamp(self, x: float, min_x: float, max_x: float) -> float:
        return max(min(x, max_x), min_x)

    def format_bounding_box(
        self,
        box: List[float],
        box_format: str = "000",
        coord_decimal: int = 3,
    ) -> str:
        box = [self.clamp(b, 0.0, 0.999) for b in box]
        if box_format == "standard":
            # NOTE: always make each coordinate 5 tokens (0.11 -> 0.110)
            box = (
                "["
                + ",".join(
                    [
                        (f"%.{coord_decimal}f" % b)[::-1].zfill(coord_decimal + 2)[::-1]
                        for b in box
                    ]
                )
                + "]"
            )
        elif box_format == "000":
            box = (
                "["
                + ",".join(
                    [
                        str(int(b * (10**coord_decimal))).zfill(coord_decimal)
                        for b in box
                    ]
                )
                + "]"
            )
        return box

    def _transform_regions(self, regions: List[Any], img_w: float, img_h: float):
        regions_out = []
        for region in regions:
            if self.region_type == "bbox":
                # region is in [x, y, w, h] format
                x, y, w, h = region
                if self.region_format == "xyxy":
                    region = [
                        x / float(img_w),
                        y / float(img_h),
                        (x + w) / float(img_w),
                        (y + h) / float(img_h),
                    ]
                elif self.region_format == "xywh":
                    region = [x / img_w, y / img_h, w / img_w, h / img_h]
                else:
                    raise ValueError(f"Unknown region format: {self.region_format}")

                # Convert boxes into string format
                region_out = self.format_bounding_box(
                    region, self.coordinate_format, self.coord_decimal
                )

                regions_out.append(region_out)
            else:
                raise ValueError(f"Unknown region type: {self.region_type}")
        return regions_out

    def _transform_conv(self, conv: str, regions: List[str]) -> str:
        if self.region_type == "bbox":
            pattern = re.compile(r"<\|bbox(\d+)\|>")
        elif self.region_type == "mask":
            pattern = re.compile(r"<\|mask(\d+)\|>")
        else:
            raise ValueError(f"Unknown region type: {self.region_type}")

        matches = pattern.finditer(conv)

        # Extract start and end indices of each match
        indices = [(match.start(), match.end()) for match in matches]

        # Replace each match with the corresponding region
        conv_out = ""
        start_idx = 0
        for i, j in indices:
            conv_out += conv[start_idx:i]

            region_idx = int(conv[i + len("<|bbox") : j - len("|>")])
            conv_out += regions[region_idx]
            start_idx = j
        conv_out += conv[start_idx:]
        return conv_out

    def __call__(
        self,
        convs: List[Dict[str, Any]],
        regions: List[Any],
        img_w: float,
        img_h: float,
    ) -> List[Dict[str, Any]]:
        # 1. transform regions
        regions = self._transform_regions(regions, img_w, img_h)

        # 2. add regions to convs
        for conv in convs:
            conv["value"] = self._transform_conv(conv["value"], regions)

        return convs
