function matrixout = vertCirc(matrixIn, numberOfRows)
matrixout = circshift(matrixIn, [numberOfRows 0]);
