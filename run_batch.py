from nbclient import NotebookClient, CellExecutionError
from nbformat import read

for nb_file in ['preprocess_images.ipynb', 'gen_STIFMaps.ipynb']:
    print(f"\n=== Running {nb_file} ===")
    try:
        with open(nb_file) as f:
            nb = read(f, as_version=4)
        client = NotebookClient(nb)
        client.execute()
        print(f"✅ Completed {nb_file}!")
    except CellExecutionError as e:
        print(f"❌ Error while running {nb_file}: {e}")
        # Uncomment next line to stop on first failure
        # break
    except Exception as e:
        print(f"❌ Unexpected error in {nb_file}: {e}")
        # break
