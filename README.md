# Viola-Jones-GPU-CPU-comparison

### Abstract

Traditionally, GPU is specifically used in display devices in computers and game consoles. Yet over the recent years, the programmability of GPU has improved significantly, being able to drive some complicated tasks other than simple graphics computations. For some computations that require high complexity, intensity and parallelism, GPU demonstrates greater advantage than CPU. Additionally, modern GPU have been designed to support advanced language, which boosts the general use of GPU, aka., GPGPU. This essay exploits the parallel logic of Viola & Jones algorithm which is commonly used in face detection and classification field, combined with GPGPU programming framework, to implement its CUDA version under GPU environment. This essay also makes analysis in regard to speed efficiency and accuracy of two different implementations, and making evaluations on GPU’s overall performance in Viola & Jones implementation.

### Keywords  

GPU programing, GPGPU, Viola & Jones, Computer Vision

### Result 
Image	| Image Width	Image Height	| # of Starting windows	| # of Detected Faces	| OpenCV running time(ms)	| CUDA running time(ms)	| Speedup VS OpenCV	| Dual GPU (ms)	| Speedup VS OpenCV	| transition
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
146_172 | 146 | 172 | 151 | 1 | 67.55 | 147.02 | 0.46 | 106.15 | 0.64 | 1.39
204_274 | 204 | 274 | 198 | 1 | 123.88 | 167.92 | 0.74 | 124.28 | 1 | 1.35
204_274 | 204 | 274 | 198 | 1 | 123.88 | 167.92 | 0.74 | 124.28 | 1 | 1.35 
233_174 | 233 | 174 | 184 | 1 | 65.08 | 128.89 | 0.5 | 96.57 | 0.67 | 1.34 
250_247 | 250 | 247 | 267 | 8 | 180.86 | 210.76 | 0.86 | 147.27 | 1.23 | 1.43 
319_467 | 319 | 467 | 369 | 1 | 288.04 | 239.14 | 1.2 | 166.86 | 1.73 | 1.44 
365_482 | 365 | 482 | 402 | 4 | 328.77 | 250.81 | 1.31 | 182.37 | 1.8 | 1.37 
463_533 | 463 | 533 | 468 | 6 | 447.28 | 304.6 | 1.47 | 205.92 | 2.17 | 1.48 
469_375 | 469 | 375 | 402 | 4 | 283.31 | 282.63 | 1 | 194.27 | 1.46 | 1.46 
500_334 | 500 | 334 | 428 | 7 | 360.82 | 270.71 | 1.33 | 197.26 | 1.83 | 1.38
500_500 | 500 | 500 | 480 | 7 | 465.29 | 336.03 | 1.38 | 227.21 | 2.05 | 1.49
512_512 | 512 | 512 | 492 | 1 | 509.61 | 333.67 | 1.53 | 226.32 | 2.25 | 1.47
550_363 | 550 | 363 | 450 | 12 | 390.8 | 290.42 | 1.35 | 214.71 | 1.82 | 1.35
627_441 | 627 | 441 | 514 | 16 | 561.15 | 350.33 | 1.6 | 230.26 | 2.44 | 1.53
670_970 | 670 | 970 | 803 | 1 | 1304.27 | 520.43 | 2.5 | 353.56 | 3.69 | 1.48
689_563 | 689 | 563 | 606 | 4 | 811.18 | 379.81 | 2.14 | 246.85 | 3.29 | 1.54
716_684 | 716 | 684 | 680 | 6 | 943.3 | 478.47 | 1.97 | 314.41 | 3 | 1.52
800_600 | 800 | 600 | 661 | 5 | 971.28 | 450.19 | 2.16 | 292.27 | 3.32 | 1.54
900_284 | 900 | 284 | 572 | 23 | 484.51 | 323.5 | 1.5 | 224.77 | 2.16 | 1.44
918_580 | 918 | 580 | 694 | 1 | 1104.78 | 513.88 | 2.15 | 342.56 | 3.23 | 1.50
1024_768 | 1024 | 768 | 928 | 1 | 1455.59 | 525.2 | 2.77 | 359.47 | 4.05 | 1.46
1152_864 | 1152 | 864 | 1107 | 1 | 2456.81 | 721.28 | 3.4 | 494.71 | 4.97 | 1.46
1212_1539 | 1212 | 1539 | 1428 | 1 | 4692.17 | 1248.29 | 3.76 | 872.15 | 5.38 | 1.43
1413_465 | 1413 | 465 | 919 | 77 | 1455.59 | 525.2 | 2.77 | 359.47 | 4.05 | 1.46
1524_1185 | 1524 | 1185 | 1307 | 1 | 3885.84 | 1058.29 | 3.67 | 698.38 | 5.56 | 1.51
1642_1022 | 1642 | 1022 | 1284 | 11 | 3478.27 | 986.17 | 3.53 | 640.13 | 5.43 | 1.54
1760_1168 | 1760 | 1168 | 1458 | 1 | 4971.46 | 1337.96 | 3.72 | 894.34 | 5.5 | 1.48
1905_1190 | 1905 | 1190 | 1560 | 1 | 5472.63 | 1439.45 | 3.8 | 977.19 | 5.6 | 1.47
2048_1536 | 2048 | 1536 | 1893 | 1 | 8156.72 | 1902.36 | 4.23 | 1357.35 | 6.01 | 1.4
22300_1601 | 2300 | 1601 | 2109 | 2 | 8991.16 | 2119.82 | 4.24 | 1471.54 | 6.11 | 1.44
2903_1664 | 2903 | 1644 | 2264 | 64 | 11686.09 | 2655.89 | 4.4 | 1820.16 | 6.42 | 1.46
2916_1587 | 2916 | 1587 | 2232 | 19 | 10970.75 | 2466.57 | 4.45 | 1697.83 | 6.46 | 1.45
|  |  |  |  |  |  |  |  |  | 1.45

### Resulting paper(in Chinese)

https://drive.google.com/open?id=0B8pACzOH-xRKNHJSMlY0RG9ob2M
