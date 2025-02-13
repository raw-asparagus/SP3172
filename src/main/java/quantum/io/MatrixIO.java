package quantum.io;

import org.ejml.data.ZMatrixRMaj;

public class MatrixIO {
    public static String prettyFormat(ZMatrixRMaj matrix) {
        StringBuilder sb = new StringBuilder();
        int rows = matrix.getNumRows();
        int cols = matrix.getNumCols();

        for (int i = 0; i < rows; i++) {
            if (i == 0) {
                sb.append("⌈\t");
            } else if (i == rows - 1) {
                sb.append("⌊\t");
            } else {
                sb.append("|\t");
            }
            for (int j = 0; j < cols; j++) {
                double real = matrix.getReal(i, j);
                double imag = matrix.getImag(i, j);
                sb.append(formatComplex(real, imag));
                if (j < cols - 1) {
                    sb.append("\t");
                }
            }
            if (i == 0) {
                sb.append("\t⌉\n");
            } else if (i == rows - 1) {
                sb.append("\t⌋");
            } else {
                sb.append("\t|\n");
            }
        }
        return sb.toString();
    }

    private static String formatComplex(double real, double imag) {
        if (imag < 0) {
            return String.format("%.3f - %.3fi", real, -imag);
        } else {
            return String.format("%.3f + %.3fi", real, imag);
        }
    }
}