"""Limpa a saída das células com a tag `dataset-images` (imagens do dataset não podem ser versionadas)."""
import sys
import nbformat as nbf

path = sys.argv[1] if len(sys.argv) > 1 else 'pratica_watermeters.ipynb'
nb = nbf.read(path, as_version=4)
n = 0
for cell in nb.cells:
    if cell.cell_type == 'code' and 'dataset-images' in cell.metadata.get('tags', []):
        cell.outputs, n = [], n + 1
nbf.write(nb, path)
print(f'{n} células com imagens do dataset limpas em {path}')
