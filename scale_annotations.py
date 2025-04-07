from globals_and_helpers import BASE_NAMES, gen_report, scale_annotations

for base_name in BASE_NAMES:
    try:
        scale_annotations(base_name, resized=True)
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
# scale_annotations("4601", resized=True)
# gen_report("7002")