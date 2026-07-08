# PRD: Clustering Recursivo Automático y Flat Renaming

## Contexto y Problema
Actualmente, el proceso de clustering genera nuevas subcarpetas y **copia** los archivos de la entrada a la salida. Esto resulta en duplicación de espacio de disco, lo cual es inaceptable para galerías masivas. 
Por otro lado, probamos que un flujo iterativo (volver a clasterizar los clusters resultantes) mejora inmensamente la calidad y granularidad de la agrupación (logrando 190+ sub-clusters muy lógicos).
Sin embargo, actualmente este proceso recursivo requiere scripts de Bash externos y mucho manejo manual.

## Objetivos (Requirements)
Este PRD detalla los cambios necesarios en el código Python (`cluster.py` / `gallery.py`) para hacer este flujo nativo, automático y eficiente en disco.

1. **Búsqueda Recursiva de Imágenes**: El script debe soportar que el directorio de entrada (`input_dir`) tenga imágenes esparcidas y anidadas en docenas de subcarpetas. El escaneo inicial de fotos debe ser **totalmente recursivo** (usando `rglob` o `os.walk`), extrayendo todo el material sin importar su nivel de anidamiento.
2. **Ciclo Automático de Iteraciones**: El script debe implementar lógica recursiva interna (hasta 3 niveles de profundidad) por defecto. No se deben usar scripts externos.
2. **Corte por Umbral (Early Stopping)**: Cualquier cluster que resulte con **menos de 15 fotos** en una iteración no debe seguir subdividiéndose. Su rama recursiva termina ahí para evitar que los algoritmos de reducción de dimensionalidad fallen por falta de muestras.
4. **MUDANZA ESTRICTA Y Flat Renaming (Ahorro de Disco)**: 
   - **PROHIBIDO COPIAR:** El script está **obligado** a mover (`shutil.move` o `os.rename`) los archivos desde sus carpetas originales a su destino. Bajo ningún concepto se pueden hacer copias (`shutil.copy`) de las imágenes originales.
   - El objetivo es desmantelar la estructura de subcarpetas de origen y consolidar todos los archivos en un único directorio plano de salida.
   - Las imágenes deben agruparse lógicamente **renombrándolas al moverlas**, usando prefijos que indiquen su linaje de clusters.
   - Ejemplo: Si `vacaciones/2023/foto_gato.jpg` cae en el cluster 2 (iteración 1), cluster 5 (iteración 2) y cluster 1 (iteración 3), su ubicación final será directa en la raíz de salida bajo el nombre `c2_c5_c1_foto_gato.jpg`.
4. **Caché de Embeddings**: Dado que el modelo procesará las mismas imágenes en las iteraciones siguientes (sólo re-calculando UMAP/Clustering sobre subconjuntos), el script debe extraer los embeddings de DINOv2/Qwen3 **una sola vez** al principio, guardarlos en memoria o en disco (`.npy`), y pasarlos a los pasos recursivos. Esto reducirá el tiempo de procesamiento a una fracción de lo que tardaría volver a procesar las imágenes.

## Especificaciones de Implementación Sugeridas

* **Refactor del pipeline en `cluster.py`**:
  - Función principal `organize_photos(input_dir, output_dir, max_iterations=3)`
  - Función de clustering que retorna las nuevas etiquetas (pero no mueve archivos aún).
  - Un bucle recursivo (o pila) que lleve la cuenta de los índices de las imágenes en cada subgrupo.
* **Procesamiento de Archivos (Fase Final)**:
  - Una vez que la estructura de prefijos de todas las iteraciones esté calculada en memoria, se hace un solo loop que recorre las fotos originales y las mueve/renombra al `output_dir`.
* **HTML Maestro Flat (`gallery.py`)**:
  - La función que genera la galería no debe leer carpetas, sino escanear el único directorio plano, parsear los nombres de los archivos por la expresión regular `^(c\d+_)+` para saber a qué "grupo" pertenece, y renderizar la interfaz agrupándolas visualmente en tarjetas/secciones sin necesidad de subcarpetas reales.

## Criterios de Aceptación
1. Al ejecutar el comando base de docker, la salida es una sola carpeta plana con imágenes renombradas.
2. El uso de disco es igual al tamaño original (no hay copias duplicadas, solo renombrado).
3. No se intenta clasterizar grupos con < 15 imágenes.
4. El archivo `index.html` procesa el nombre de los archivos y muestra visualmente los grupos separados.
5. El sistema encuentra todas las fotos en la entrada sin importar cuán anidadas estén en múltiples subcarpetas.
6. El código **fuerza estrictamente una operación de MOVE (desplazar)**, destruyendo/vaciando la jerarquía de origen y asegurando que 10GB de fotos originales no se conviertan en 20GB tras el proceso.
