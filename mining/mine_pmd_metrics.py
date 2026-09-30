"""Static metrics: count PMD violations per file from a PMD XML report."""
import xml.etree.ElementTree as ET

import pandas as pd


def parse_pmd_report(xml_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for f in root.iter():
        if f.tag.endswith("file"):
            rows.append({"file": f.attrib["name"],
                         "pmd_violations": sum(1 for v in f if v.tag.endswith("violation"))})
    return pd.DataFrame(rows, columns=["file", "pmd_violations"])
