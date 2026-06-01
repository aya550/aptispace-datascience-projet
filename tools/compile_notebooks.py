import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid

# Mode logs : exécuter les .py et capturer stdout/stderr (lent — optionnel)
RUN_LOGS = "--with-logs" in sys.argv

# Noms locaux introduits par des imports de packages absents (ex: 'tf', 'layers')
# Partagé sur toute la session Quarto (un seul kernel pour tous les notebooks)
_unavailable_names: set = set()


def _parse_import_names(s: str) -> list:
    """Extrait les noms locaux introduits par une ligne d'import."""
    names = []
    if s.startswith('import '):
        for part in s[7:].split(','):
            part = part.strip()
            alias = part.split(' as ')[-1].strip().split('.')[0]
            names.append(alias)
    elif s.startswith('from ') and ' import ' in s:
        import_part = s.split(' import ', 1)[1].strip()
        if import_part != '*':
            for part in import_part.split(','):
                part = part.strip()
                alias = part.split(' as ')[-1].strip()
                if alias.isidentifier():
                    names.append(alias)
    return names


def _wrap_missing_imports(source_lines: list) -> list:
    """
    Remplace les lignes d'import de packages absents par try/except ImportError: pass.
    Collecte aussi les noms importés depuis ces packages dans _unavailable_names.
    """
    global _unavailable_names
    result = []
    for line in source_lines:
        s = line.strip()
        pkg = None
        if s.startswith('import ') and not s.startswith('#'):
            pkg = s.split()[1].split('.')[0]
        elif s.startswith('from ') and ' import' in s and not s.startswith('#'):
            pkg = s.split()[1].split('.')[0]

        if pkg and ',' not in pkg and importlib.util.find_spec(pkg) is None:
            _unavailable_names.update(_parse_import_names(s))
            indent = ' ' * (len(line) - len(line.lstrip()))
            result.append(f"{indent}try:\n")
            result.append(f"{indent}    {s}\n")
            result.append(f"{indent}except ImportError:\n")
            result.append(f"{indent}    pass\n")
        else:
            result.append(line)
    return result


def _cell_uses_unavailable(source_lines: list) -> bool:
    """
    Retourne True si la cellule (hors lignes d'import) référence un nom
    introduit par un package absent — ces cellules doivent être #| eval: false.
    """
    if not _unavailable_names:
        return False
    for line in source_lines:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('import ') or s.startswith('from '):
            continue
        for name in _unavailable_names:
            if re.search(r'\b' + re.escape(name) + r'\b', s):
                return True
    return False


# ---------------------------------------------------------------------------
# Répertoires de travail
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
notebooks_dir = os.path.join(base_dir, 'notebooks')

# Rendre les modules locaux du projet (src/, etc.) trouvables par find_spec()
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

build_dir = os.path.join(base_dir, 'build')
src_dir = os.path.join(build_dir, 'src')
logs_dir = os.path.join(build_dir, 'logs')
report_notebooks_dir = os.path.join(build_dir, 'notebooks')

os.makedirs(src_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(report_notebooks_dir, exist_ok=True)

notebook_files = sorted([f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')])

print(f"🚀 Début de la compilation de {len(notebook_files)} notebooks...")

for nb_file in notebook_files:
    nb_path = os.path.join(notebooks_dir, nb_file)
    name_no_ext = os.path.splitext(nb_file)[0]

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    py_lines = []
    qmd_lines = []

    py_lines.append("import os, sys")
    py_lines.append(f"sys.path.append(r'{base_dir}')\n")

    for cell in nb_data.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])

        if isinstance(source, str):
            source = source.splitlines(keepends=True)

        source_str = "".join(source)

        if cell_type == 'markdown':
            qmd_lines.append(source_str + "\n\n")

        elif cell_type == 'code':
            if not source_str.strip():
                continue

            cell_outputs = cell.get('outputs', [])

            source_has_eval_directive = any(
                line.strip().startswith('#| eval:') for line in source
            )
            # Toujours désactiver l'exécution : les outputs sont pré-embarqués dans le QMD
            skip_eval = not source_has_eval_directive

            fixed_source = _wrap_missing_imports(source)
            fixed_source_str = "".join(fixed_source)

            # --- Bloc QMD (code affiché, non ré-exécuté si outputs présents) ---
            qmd_lines.append("```{python}\n")
            if skip_eval:
                qmd_lines.append("#| eval: false\n")
            qmd_lines.append("#| output: false\n")
            qmd_lines.append(fixed_source_str)
            if not fixed_source_str.endswith("\n"):
                qmd_lines.append("\n")
            qmd_lines.append("```\n\n")

            # --- Embed des outputs pré-calculés ---
            for output in cell_outputs:
                otype = output.get('output_type', '')
                data  = output.get('data', {})

                # Images matplotlib/seaborn (PNG base64)
                if 'image/png' in data:
                    png = data['image/png']
                    if isinstance(png, list):
                        png = ''.join(png)
                    qmd_lines.append(f'![](data:image/png;base64,{png.strip()})\n\n')

                # Plotly JSON interactif
                elif 'application/vnd.plotly.v1+json' in data:
                    plotly_data = data['application/vnd.plotly.v1+json']
                    fig_data   = json.dumps(plotly_data.get('data', []))
                    fig_layout = json.dumps(plotly_data.get('layout', {}))
                    fig_config = json.dumps(plotly_data.get('config', {}))
                    div_id     = f"plotly-{uuid.uuid4()}"
                    plotly_html = (
                        f'<div id="{div_id}" style="width:100%;height:400px;"></div>\n'
                        f'<script>document.addEventListener("DOMContentLoaded",function(){{'
                        f'if(typeof Plotly!=="undefined")Plotly.newPlot("{div_id}",{fig_data},{fig_layout},{fig_config});}});</script>\n'
                    )
                    qmd_lines.append('::: {.content-visible unless-format="typst"}\n')
                    qmd_lines.append(plotly_html)
                    qmd_lines.append(":::\n\n")

                # HTML Plotly alternatif
                elif 'text/html' in data:
                    html_lines   = data['text/html']
                    html_content = "".join(html_lines) if isinstance(html_lines, list) else html_lines
                    if 'plotly' in html_content.lower() or 'dataframe' in html_content.lower():
                        qmd_lines.append('::: {.content-visible unless-format="typst"}\n')
                        qmd_lines.append(html_content)
                        if not html_content.endswith("\n"):
                            qmd_lines.append("\n")
                        qmd_lines.append(":::\n\n")

                # Sorties texte (print, stdout)
                elif otype == 'stream' and output.get('name') == 'stdout':
                    text = ''.join(output.get('text', []))
                    if text.strip():
                        qmd_lines.append(f'```\n{text.rstrip()}\n```\n\n')

                # text/plain simple (résultats)
                elif 'text/plain' in data and otype == 'execute_result':
                    text = ''.join(data['text/plain']) if isinstance(data['text/plain'], list) else data['text/plain']
                    if text.strip() and not text.strip().startswith('<'):
                        qmd_lines.append(f'```\n{text.rstrip()}\n```\n\n')

            # --- Script .py (commentaires sur les magics IPython) ---
            py_cell_lines = []
            for line in source:
                stripped = line.strip()
                if stripped.startswith('%') or stripped.startswith('!'):
                    py_cell_lines.append(f"# {line}")
                else:
                    py_cell_lines.append(line)
            py_cell_str = "".join(py_cell_lines)
            py_lines.append(py_cell_str)
            if not py_cell_str.endswith("\n"):
                py_lines.append("\n")

    # Écriture .py
    py_file_path = os.path.join(src_dir, f"{name_no_ext}.py")
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(py_lines))
    print(f"  ➡️  [PY]  Généré : {py_file_path}")

    # Écriture .qmd
    qmd_file_path = os.path.join(report_notebooks_dir, f"{name_no_ext}.qmd")
    with open(qmd_file_path, 'w', encoding='utf-8') as f:
        f.write("".join(qmd_lines))
    print(f"  ➡️  [QMD] Généré : {qmd_file_path}")

    # Exécution .log (optionnel)
    log_file_path = os.path.join(logs_dir, f"{name_no_ext}.log")
    if RUN_LOGS:
        try:
            result = subprocess.run(
                [sys.executable, py_file_path],
                capture_output=True, text=True,
                env={**os.environ, "MPLBACKEND": "agg"},
                cwd=notebooks_dir,
                timeout=30,
            )
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            print(f"  ➡️  [LOG] Généré : {log_file_path} (Exit Code: {result.returncode})")
        except subprocess.TimeoutExpired:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write("❌ ÉCHEC : Temps d'exécution limite dépassé (30s).")
            print(f"  ➡️  [LOG] Généré : {log_file_path} (TIMEOUT)")
        except Exception as e:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"❌ ÉCHEC : Erreur lors de l'exécution : {e}")
            print(f"  ➡️  [LOG] Généré : {log_file_path} (ERROR)")
    else:
        print(f"  ⏭️  [LOG] Ignoré (utiliser --with-logs pour activer)")

print("✅ Compilation des notebooks terminée !")
