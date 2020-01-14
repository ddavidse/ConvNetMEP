function matrixout = horCirc(matrixIn, numberOfRows)
matrixout = circshift(matrixIn, [0 numberOfRows]);
