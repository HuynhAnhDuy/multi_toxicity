from lxml import etree
import os

# Đường dẫn tới thư mục chứa 8 file .xml
xml_dir = "features"

# Duyệt qua tất cả file .xml trong thư mục
for filename in os.listdir(xml_dir):
    if filename.endswith(".xml"):
        path = os.path.join(xml_dir, filename)
        tree = etree.parse(path)
        root = tree.getroot()

        # Tìm và bật tất cả descriptor
        for descriptor in root.xpath("//Descriptor"):
            descriptor.set("value", "true")

        # Ghi đè lại file
        tree.write(path, pretty_print=True, encoding="UTF-8", xml_declaration=True)

        print(f"✅ Updated: {filename}")
