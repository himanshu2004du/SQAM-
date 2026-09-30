"""Coverage metrics: line coverage per source file from a JaCoCo XML report."""
import xml.etree.ElementTree as ET

import pandas as pd


def parse_jacoco_report(xml_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for pkg in root.iter("package"):
        for sf in pkg.iter("sourcefile"):
            for c in sf.findall("counter"):
                if c.attrib["type"] == "LINE":
                    missed, covered = int(c.attrib["missed"]), int(c.attrib["covered"])
                    tot = missed + covered
                    rows.append({"file": f"{pkg.attrib['name']}/{sf.attrib['name']}",
                                 "line_coverage": covered / tot if tot else 0.0})
    return pd.DataFrame(rows, columns=["file", "line_coverage"])
