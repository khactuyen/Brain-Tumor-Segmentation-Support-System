# Template execution contract

## Reference

- Retained source: `D:\SIC_Capstone 2026\doc\SIC_AI_Capstone Project_Final Report.docx`
- SHA-256: `2541B2276482A642B9128013C2512F8A964E19A6D0DAE09E65E262DE2F89AD89`
- Saved page count: 7 pages
- Section count: 1
- Evidence: `template-reference.pdf`, `template-reference-render/page-1.png` through `page-7.png`, and `template-style-evidence.json` in this directory
- Reference purpose: visual and structural authority for a new, expanded capstone report

## Page system

- ISO A4 portrait, 8.27 by 11.69 inches
- Margins: left 1.00 inch, right 1.00 inch, top 1.18 inch, bottom 1.00 inch
- One column
- Different first page enabled
- Header linked to the source section; footer uses PAGE and NUMPAGES fields
- Reference has one section. The expanded report may add section breaks only to manage front matter and appendices while retaining A4 geometry.

## Typography and color

- Primary families embedded by the template: SamsungOne 400, SamsungOne 700, and Samsung Sharp Sans
- Body fallback for Vietnamese glyph coverage: Arial, 10.5 to 11 pt
- Main report title: source-derived Samsung blue, approximately `#2146B7`, large bold type
- Chapter bars: Samsung blue fill with white bold text
- Section headings: black bold text; no decorative rules below titles
- Body: black, left aligned or justified, 1.12 to 1.18 line spacing, 6 pt after paragraphs
- Captions: 9 pt, centered, black
- Footer: right aligned page number in source-derived style

## Lists and tables

- Tables use light gray `#D9D9D9` borders, dark-blue or light-blue header fills, and generous cell padding.
- Short numeric columns are centered. Narrative columns are left aligned.
- Header rows repeat across pages when a table spans more than one page.
- No fixed row heights that can clip text.
- Bullets are used for distinct requirements, limitations, and planned actions; connected analysis remains prose.

## Recurring components

- Cover pattern: Samsung logo at upper left; course label; large blue report title; audience line; copyright block; source footer logo.
- Project identity pattern: project title, date, team name, and five member slots.
- Content pattern: large blue `Content` heading and chapter list.
- Chapter pattern: full-width blue bar with white chapter title.
- Review pattern: team picture placeholder, member review table, and instructor scoring table.
- Footer pattern: `Page X / Y` aligned at the lower right.

## Content flow

1. Cover
2. Project identity
3. Vietnamese summary and English abstract
4. Contents, figures, tables, and abbreviations
5. Introduction
6. Project execution
7. Results
8. Projected impact
9. Team member review
10. Instructor review
11. References
12. Technical appendices

## Slot map

- Cover title: rewrite for the brain tumor segmentation support system.
- Project title and date: rewrite with the project title and document date.
- Team name and member slots: use `AI Healthcare Team` and visible `[BỔ SUNG]` placeholders because the source material does not identify members.
- Contents page: replace with a generated multi-page static contents list; set `w:updateFields=true` for Word refresh.
- Pages 4 and 5 source section placeholders: replace with the full expanded report body.
- Team picture slot: retain as a visible `[BỔ SUNG ẢNH NHÓM]` placeholder.
- Team review rows: retain five editable rows with placeholders.
- Instructor review: preserve the source categories and blank scoring/comment fields.

## Package preservation

- Preserve-only: embedded fonts, `word/theme/theme1.xml`, `word/fontTable.xml`, font relationships, `word/media/image1.png`, headers, footer logo, custom XML, and package-level relationships unless a required body relationship is added.
- Editable: `word/document.xml`, styles needed for added semantic headings, numbering needed for added lists, settings for field refresh, footer PAGE and NUMPAGES cache text, and document properties.
- The original retained DOCX remains byte-for-byte unchanged.
- The working copy may be rewritten by python-docx because the user explicitly requires more than 50 pages, which cannot fit the source content controls. The visual system and review tables must remain recognizably source-derived.

## Fidelity gates

- Reference SHA-256 must remain unchanged.
- Final document remains A4 portrait with the same margin proportions and Samsung blue visual language.
- Cover, project identity, contents, chapter bars, footer, and review tables must remain recognizably derived from the reference.
- Final report must exceed 50 rendered pages without using blank filler pages.
- All pages must render without clipping, overlapping, broken tables, missing Vietnamese glyphs, or misplaced headers and footers.
- Results must reproduce notebook outputs exactly and identify the mismatch between v2 and v3 evaluation protocols.
- No claim of clinical readiness or statistically fair model superiority is allowed.
