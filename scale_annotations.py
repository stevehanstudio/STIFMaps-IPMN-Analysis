from globals_and_helpers import BASE_NAMES, gen_report, scale_annotations

for base_name in BASE_NAMES:
    scale_annotations(base_name, resized=True)
# scale_annotations("4601")
# gen_report("7002")