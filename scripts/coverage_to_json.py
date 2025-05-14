import json
import sys
import xml.etree.ElementTree as ET


def main(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # coverage.xml root has line-rate attribute in <coverage>
    rate = float(root.get('line-rate', 0)) * 100
    pct = round(rate, 1)
    if pct >= 90:
        color = 'brightgreen'
    elif pct >= 75:
        color = 'yellow'
    else:
        color = 'red'
    badge = {
        'schemaVersion': 1,
        'label': 'coverage',
        'message': f'{pct}%',
        'color': color
    }
    print(json.dumps(badge))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python coverage_to_json.py <coverage.xml>', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
