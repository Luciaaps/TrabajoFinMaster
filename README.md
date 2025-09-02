# Modelos de difusión para la generación de imágenes médicas: aplicación al cáncer oral
## Resumen
En los últimos años, los modelos de difusión han revolucionado la inteligencia artificial, especialmente en la generación de imágenes realistas. Sin embargo, su aplicación en imagen médica sigue siendo un desafío, a pesar de que la generación de imágenes sintéticas de alta calidad puede ser clave para avanzar en el diagnóstico y tratamiento de enfermedades con conjuntos de datos limitados, como el cáncer oral.

Este trabajo explora la aplicabilidad de distintos modelos de difusión para generar imágenes médicas de cáncer oral. Se emplea una base de datos de imágenes RGB de pacientes en diferentes estadios de la enfermedad, y se implementan, entrenan y comparan varios enfoques de generación para obtener imágenes sintéticas representativas tanto de fases iniciales como avanzadas.

El estudio incluye la normalización y preprocesamiento de los datos, la aplicación de modelos de difusión y la evaluación de la calidad de las imágenes generadas mediante métricas cuantitativas como el \textit{Fréchet Inception Distance} (FID). Asimismo, se analiza el impacto del uso de estas imágenes sintéticas en el rendimiento de un clasificador de estadios. Además, se evalúa el efecto de emplear imágenes segmentadas mediante un recorte cuadrado centrado en la lesión sobre la calidad de la generación y los resultados del clasificador.

Los resultados muestran que la aumentación clásica (F1-test de 0.680) y el modelo DDIM (F1-test de 0.615) ofrecen los mejores resultados globales, logrando una mejora en la clasificación de estadios. Además, el modelo DDIM presenta la mejor calidad de imágenes generadas. En contraste, la estrategia basada en segmentación cuadrada no alcanza el mismo nivel de estabilidad, lo que se refleja en valores FID elevados en estadios avanzados, afectando negativamente el rendimiento del clasificador.

## Abstract
In recent years, diffusion models have revolutionized artificial intelligence, especially in the generation of realistic images. However, their application in medical imaging remains a challenge, despite the fact that generating high-quality synthetic images can be key to advancing the diagnosis and treatment of diseases with limited datasets, such as oral cancer.

This work explores the applicability of different diffusion models to generate medical images of oral cancer. A database of RGB images of patients at different stages of the disease is used, and several generation approaches are implemented, trained, and compared to obtain synthetic images representative of both early and advanced phases.

The study includes data normalization and preprocessing, the application of diffusion models, and the evaluation of the quality of generated images using quantitative metrics such as the \textit{Fréchet Inception Distance} (FID). The impact of using these synthetic images on the performance of a stage classifier is also analyzed. In addition, the effect of using segmented images through a square crop centered on the lesion on both generation quality and classifier results is assessed.

The results show that classical augmentation (F1-test of 0.680) and the DDIM model (F1-test of 0.615) provide the best overall outcomes, achieving an improvement in stage classification. Moreover, the DDIM model delivers the highest quality in generated images. In contrast, the strategy based on square segmentation does not reach the same level of stability, which is reflected in high FID values in advanced stages, negatively impacting classifier performance.

## Resum
En els últims anys, els models de difusió han revolucionat la intel·ligència artificial, especialment en la generació d’imatges realistes. No obstant això, la seua aplicació en imatge mèdica continua sent un repte, malgrat que la generació d’imatges sintètiques d’alta qualitat pot ser clau per avançar en el diagnòstic i tractament de malalties amb conjunts de dades limitats, com el càncer oral.

Aquest treball explora l’aplicabilitat de diferents models de difusió per generar imatges mèdiques de càncer oral. S’empra una base de dades d’imatges RGB de pacients en diferents estadis de la malaltia, i s’implementen, entrenen i comparen diversos enfocaments de generació per obtindre imatges sintètiques representatives tant de fases inicials com avançades.

L’estudi inclou la normalització i preprocessament de les dades, l’aplicació de models de difusió i l’avaluació de la qualitat de les imatges generades mitjançant mètriques quantitatives com el \textit{Fréchet Inception Distance} (FID). Així mateix, s’analitza l’impacte de l’ús d’aquestes imatges sintètiques en el rendiment d’un classificador d’estadis. A més, s’avalua l’efecte d’emprar imatges segmentades mitjançant un retall quadrat centrat en la lesió sobre la qualitat de la generació i els resultats del classificador.

Els resultats mostren que l’augmentació clàssica (F1-test de 0,680) i el model DDIM (F1-test de 0,615) ofereixen els millors resultats globals, aconseguint una millora en la classificació d’estadis. A més, el model DDIM presenta la millor qualitat d’imatges generades. En canvi, l’estratègia basada en segmentació quadrada no aconsegueix el mateix nivell d’estabilitat, cosa que es reflecteix en valors FID elevats en estadis avançats, afectant negativament el rendiment del classificador.
