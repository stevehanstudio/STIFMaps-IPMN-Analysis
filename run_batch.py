from nbconvert.preprocessors import CellExecutionError
from nbclient import NotebookClient
from nbformat import read, write

for nb_file in ['preprocess_images.ipynb', 'gen_STIFMaps.ipynb']:
    print(f"\n=== Running {nb_file} ===")
    try:
        with open(nb_file) as f:
            nb = read(f, as_version=4)

        client = NotebookClient(nb)
        client.execute()

        # Display outputs on the terminal
        for cell in nb['cells']:
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        print(output['text'], end="")
                    elif 'data' in output and 'text/plain' in output['data']:
                        print(output['data']['text/plain'], end="")

        # Save the executed notebook with outputs
        executed_file = f"executed_{nb_file}"
        with open(executed_file, 'w') as f_out:
            write(nb, f_out)
        print(f"\n✅ Completed {nb_file}! Outputs saved to {executed_file}")

    except CellExecutionError as e:
        print(f"❌ Error while running {nb_file}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in {nb_file}: {e}")
