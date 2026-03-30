# Notes

# Resources / Apps

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

