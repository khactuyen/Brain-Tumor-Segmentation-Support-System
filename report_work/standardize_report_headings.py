"""Surgically update report headings and its cached Word TOC entries."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


REPORT = Path(
    r"D:\SIC_Capstone 2026\doc\SIC_AI_Capstone Project_Final Report_Brain Tumor Segmentation_Revised.updated.final.docx"
)
OUTPUT = REPORT.with_name(REPORT.stem + ".vi.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}

TRANSLATIONS = {
    "PROJECT INFORMATION": "THÔNG TIN DỰ ÁN",
    "TABLE OF CONTENTS": "MỤC LỤC",
    "LISTS AND ABBREVIATIONS": "DANH MỤC HÌNH, BẢNG VÀ CHỮ VIẾT TẮT",
    "EXECUTIVE SUMMARY": "TÓM TẮT ĐIỀU HÀNH",
    "ABSTRACT": "TÓM TẮT",
    "Introduction": "Giới thiệu",
    "Background Information": "Thông tin bối cảnh",
    "Motivation and Objective": "Động lực và mục tiêu",
    "Discover và bằng chứng đã sử dụng": "Giai đoạn khám phá và bằng chứng đã sử dụng",
    "Persona giả định và giả thuyết cần kiểm chứng": "Chân dung người dùng giả định và giả thuyết cần kiểm chứng",
    "Members and Role Assignments": "Phân công thành viên và vai trò",
    "Schedule and Milestones": "Tiến độ và các mốc thực hiện",
    "Project Execution": "Triển khai dự án",
    "Data Acquisition": "Thu thập dữ liệu",
    "Training Methodology": "Phương pháp huấn luyện",
    "Kiến trúc 3D U-Net": "Kiến trúc 3D U-Net và Swin UNETR",
    "Residual unit InstanceNorm và dropout": "Khối residual, InstanceNorm và dropout",
    "Decoder và skip connection": "Bộ giải mã và kết nối tắt (skip connection)",
    "Scaled dot product self attention": "Self-attention tích vô hướng có điều chỉnh tỉ lệ",
    "Window attention và shifted window": "Attention theo cửa sổ và cửa sổ dịch chuyển",
    "Loss, optimizer và lịch học": "Hàm mất mát, bộ tối ưu và lịch điều chỉnh tốc độ học",
    "Logit softmax và xác suất voxel": "Logit, softmax và xác suất voxel",
    "Dice Loss": "Hàm mất mát Dice",
    "Cross Entropy": "Hàm mất mát cross-entropy",
    "Learning rate và cosine annealing": "Tốc độ học và cosine annealing",
    "Early stopping và checkpoint selection": "Dừng sớm và lựa chọn checkpoint",
    "Sliding-window inference": "Suy luận bằng cửa sổ trượt",
    "Macro micro và conditional metric": "Chỉ số macro, micro và có điều kiện",
    "Từ công thức đến protocol đánh giá": "Từ công thức đến quy trình đánh giá",
    "Workflow": "Quy trình làm việc",
    "System Design": "Thiết kế hệ thống",
    "Results": "Kết quả",
    "Data Preprocessing": "Tiền xử lý dữ liệu",
    "Exploratory Data Analysis (EDA)": "Phân tích khám phá dữ liệu (EDA)",
    "EDA notebook v2": "Phân tích EDA của notebook v2",
    "Modeling – Kết quả 3D U-Net": "Kết quả mô hình 3D U-Net và Swin UNETR",
    "So sánh chỉ số as-reported": "So sánh chỉ số theo số liệu được báo cáo",
    "User Interface": "Giao diện người dùng",
    "Testing and Improvements": "Kiểm thử và cải tiến",
    "Kế hoạch usability test": "Kế hoạch kiểm thử khả dụng",
    "Projected Impact": "Tác động dự kiến",
    "Accomplishments and Benefits": "Thành quả và lợi ích",
    "Future Improvements": "Hướng cải tiến trong tương lai",
    "Quy trình follow-up vấn đề": "Quy trình theo dõi và xử lý vấn đề",
    "Team Member Review and Comment": "Đánh giá và nhận xét thành viên nhóm",
    "Retrospective của nhóm": "Tổng kết rút kinh nghiệm của nhóm",
    "Instructor Review and Comment": "Đánh giá và nhận xét của giảng viên",
    "General comment": "Nhận xét chung",
    "REFERENCES": "TÀI LIỆU THAM KHẢO",
    "APPENDICES": "PHỤ LỤC",
    "Appendix A – Cấu hình tái lập notebook v1": "Phụ lục A – Cấu hình tái lập notebook v1",
    "Appendix B – Cấu hình tái lập notebook v2": "Phụ lục B – Cấu hình tái lập notebook v2",
    "Appendix C – Định nghĩa chỉ số và empty-mask policy": "Phụ lục C – Định nghĩa chỉ số và quy tắc xử lý mặt nạ rỗng",
    "Appendix D – Ma trận truy vết yêu cầu": "Phụ lục D – Ma trận truy vết yêu cầu",
    "Appendix E – Danh sách test case chi tiết": "Phụ lục E – Danh sách ca kiểm thử chi tiết",
    "Appendix F – Model card tóm tắt 3D U-Net": "Phụ lục F – Bản mô tả mô hình 3D U-Net",
    "Appendix G – Model card tóm tắt Swin UNETR": "Phụ lục G – Bản mô tả mô hình Swin UNETR",
    "Appendix H – Hướng dẫn sử dụng nhanh": "Phụ lục H – Hướng dẫn sử dụng nhanh",
    "Appendix I – Bản đồ artifact và khả năng tái lập": "Phụ lục I – Danh mục thành phần dự án và khả năng tái lập",
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(node.text or "" for node in paragraph.xpath(".//w:t", namespaces=NS))


def replace_text(paragraph: etree._Element, new_text: str, *, toc: bool) -> None:
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not nodes:
        raise ValueError(f"Paragraph has no text node: {new_text}")
    if toc:
        # The final text node is the cached PAGEREF page number.
        title_nodes = nodes[:-1]
        if not title_nodes:
            raise ValueError(f"TOC entry has no title node: {new_text}")
        nodes = title_nodes
    nodes[0].text = new_text
    for node in nodes[1:]:
        node.text = ""


def revised_heading(text: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", text)
    number, title = match.groups() if match else ("", text)
    if number.startswith("2.2.4."):
        number = number.replace("2.2.4.", "2.2.3.", 1)
    else:
        major = re.fullmatch(r"2\.2\.([5-9])(\..+)?", number)
        if major:
            number = f"2.2.{int(major.group(1)) - 1}{major.group(2) or ''}"
    title = TRANSLATIONS.get(title, title)
    return f"{number} {title}" if number else title


def main() -> None:
    with ZipFile(REPORT) as source:
        xml = source.read("word/document.xml")
        root = etree.fromstring(xml)
        paragraphs = root.xpath("//w:body/w:p", namespaces=NS)
        changed = []
        removed = []
        for paragraph in paragraphs:
            style = paragraph.find("./w:pPr/w:pStyle", namespaces=NS)
            if style is None:
                continue
            style_name = style.get(f"{{{W}}}val", "")
            is_heading = style_name.startswith("Heading")
            is_toc = style_name.lower().startswith("toc")
            if not (is_heading or is_toc):
                continue
            old = paragraph_text(paragraph)
            if is_toc:
                # Cached TOC paragraph also includes a final page number.
                text_nodes = paragraph.xpath(".//w:t", namespaces=NS)
                old = "".join(node.text or "" for node in text_nodes[:-1])
            if old == "2.2.4 Kiến trúc Swin UNETR":
                paragraph.getparent().remove(paragraph)
                removed.append((style_name, old))
                continue
            new = revised_heading(old)
            if new != old:
                replace_text(paragraph, new, toc=is_toc)
                changed.append((style_name, old, new))

        heading_changes = [item for item in changed if item[0].startswith("Heading")]
        toc_changes = [item for item in changed if item[0].lower().startswith("toc")]
        if len(removed) != 2 or len(heading_changes) < 40 or len(toc_changes) < 40:
            raise RuntimeError(
                f"Unexpected edit scope: {len(removed)} removals, "
                f"{len(heading_changes)} headings, {len(toc_changes)} TOC entries"
            )

        settings = etree.fromstring(source.read("word/settings.xml"))
        update_fields = settings.find(f"{{{W}}}updateFields")
        if update_fields is None:
            update_fields = etree.SubElement(settings, f"{{{W}}}updateFields")
        update_fields.set(f"{{{W}}}val", "true")

        with tempfile.NamedTemporaryFile(
            suffix=".docx", prefix="heading_edit_", dir=REPORT.parent, delete=False
        ) as handle:
            staged = Path(handle.name)
        try:
            with ZipFile(staged, "w") as output:
                for info in source.infolist():
                    payload = source.read(info.filename)
                    if info.filename == "word/document.xml":
                        payload = etree.tostring(root, encoding="UTF-8", xml_declaration=True)
                    elif info.filename == "word/settings.xml":
                        payload = etree.tostring(settings, encoding="UTF-8", xml_declaration=True)
                    output.writestr(info, payload)
        except Exception:
            staged.unlink(missing_ok=True)
            raise

    try:
        os.replace(staged, OUTPUT)
    finally:
        staged.unlink(missing_ok=True)

    print(f"Saved {OUTPUT}")
    print(f"Updated {len(heading_changes)} headings and {len(toc_changes)} cached TOC entries")
    print("Removed:", removed)
    for _, old, new in heading_changes:
        print(f"{old} -> {new}")


if __name__ == "__main__":
    main()
