from globals_and_helpers import (
    BASE_NAMES,
    filter_measurements,
)

for base_name in BASE_NAMES:
    try:
        filter_measurements(base_name)
    except Exception as e:
        print(f"Error processing {base_name}: {e}")

# filter_measurements("27620")
# filter_measurements("4601")
# filter_measurements("7002")

