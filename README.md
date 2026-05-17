# Notes

# Resources / Apps

## Entorno Python (Poetry)

### Requisitos previos

- **Python 3.11+** — se recomienda usar [pyenv](https://github.com/pyenv/pyenv) para gestionar versiones.
- **Poetry 1.8+** — gestor de dependencias y entornos virtuales.

```bash
# Instalar pyenv (si no lo tienes)
brew install pyenv

# Instalar Python 3.12
pyenv install 3.12.9
pyenv local 3.12.9   # crea .python-version en el proyecto

# Instalar Poetry (si no lo tienes)
curl -sSL https://install.python-poetry.org | python3 -
```

### Montar el entorno

```bash
# Clonar el repositorio y entrar al directorio
git clone <repo-url>
cd university-notes

# Instalar todas las dependencias (crea el virtualenv automáticamente)
poetry install

# Activar el entorno en la shell actual
poetry shell
```

> **Nota:** Poetry crea el virtualenv dentro del proyecto si configuras `poetry config virtualenvs.in-project true`, lo que facilita que VS Code lo detecte automáticamente.

### Abrir JupyterLab

Con el entorno activo:

```bash
poetry run jupyter lab
```

O si ya ejecutaste `poetry shell`:

```bash
jupyter lab
```

Esto abre JupyterLab en el navegador en `http://localhost:8888`. Para especificar un directorio de trabajo distinto:

```bash
poetry run jupyter lab --notebook-dir=./semester_1
```

---

## Configuración de LaTeX en Visual Studio Code

Este repositorio contiene la configuración necesaria para trabajar con documentos LaTeX de manera profesional utilizando **Visual Studio Code (VS Code)** como IDE principal.

---

### 🚀 Requisitos Previos

Para que VS Code pueda compilar documentos, primero debes instalar una distribución de LaTeX en tu sistema operativo:

#### 🪟 Windows
1. Descarga e instala **[MiKTeX](https://miktex.org/download)** (opción recomendada por su gestor de paquetes automático).
2. Durante la instalación, selecciona la opción *"Install missing packages on the fly: Yes"* para evitar interrupciones al compilar.
3. Reinicia VS Code después de la instalación.

#### 🍎 macOS
1. Descarga e instala **[MacTeX](https://www.tug.org/mactex/)** (la distribución completa).
2. Verifica que los comandos de LaTeX estén en tu PATH (generalmente se configura solo).

---

### 🛠️ Configuración de VS Code

#### 1. Extensión Principal
Instala la extensión **[LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)** desde el Marketplace de VS Code. Esta herramienta habilita:
* Compilación automática al guardar (`Ctrl+S` / `Cmd+S`).
* Visor de PDF integrado con sincronización directa (SyncTeX).
* IntelliSense para comandos y citas bibliográficas.

#### 2. Configuración del Proyecto (`settings.json`)
Para mantener el folder limpio de archivos auxiliares (`.aux`, `.log`, `.out`, etc.), se recomienda usar una carpeta de salida. 

1. Presiona `Ctrl+Shift+P` (Win) o `Cmd+Shift+P` (Mac).
2. Busca: **Preferences: Open User Settings (JSON)**.
3. Pega la siguiente configuración:

```json
{
    "latex-workshop.latex.outDir": "%DIR%/build",
    "latex-workshop.latex.clean.enabled": true,
    "latex-workshop.view.pdf.viewer": "tab",
    "latex-workshop.latex.autoBuild.run": "onSave"
}
```

