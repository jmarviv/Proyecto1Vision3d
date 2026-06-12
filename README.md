# PASOS PARA PODER INICIAR EL FUNCIONAMIENTO DEL CÓDIGO:

1) MEDIR LOS ARUCOS Y ESCRIBIRLOS EN EL MAIN:

Dentro de este proyecto es muy importante la precisión, por ende, tenemos que medir
con claridad cada esquina d los arucos al centro del eje [0,0] y apuntarla en la primera linea del main
y de calibracion.py

Es importante saber que la medida usada en este proyectos es en METROS, por ende esta distancia
debe de ser apuntada en dichas unidades.


2) INICIAR UN SERVIDOR HTTP DE CÁMARA A TRAVÉS DEL MÓVIL:

En mi caso he utilizado una aplicación del móvil llamada IP webcam, que habre un puerto e inicia un
servidor. Para que esto funcione, el móvil y el ordenador han de estar conectados a la misma red,
preferiblemente si se comparten datos del móvil al ordenador.
Tras haber entrado a dicha aplicación, saldrán una lista de opciones, se ha de buscar la opción
"Iniciar servidor", al elegirla, se abrirá la cámara con una url, apunta esa url en el main.


3) EJECUTAR EL MAIN:

Esto es simple, se han de tener todos los ficheros incluidos en el repositorio ubicados en la misma
carpeta y ejecutar el fichero main


4) REALIZAR LA CALIBRACIÓN:

Tras iniciar el main, comenzará la calibración, la cual está explicada en la memoria, pero lo que hay
que saberes que, cada 0,6 segundos, toma una foto SI SE DETECTAN LOS 4 ARUCOS hasta que llegue a las 40
(dicho número se puede cambiar dentro del código calibracion.py).
Por ende, es importante ir moviendo la cámara para que tome esas fotos desde ángulos diferentes pero
siempre mostrando los arucos


5) DISFRUTAR
