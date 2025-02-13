package quantum.basis;

import org.ejml.data.ZMatrixRMaj;
import org.ejml.dense.row.CommonOps_ZDRM;

import quantum.io.MatrixIO;

public class CatBasis extends Basis {

    private final int N;    // Fock space dimension for encoding
    private final double[] alpha;

    public CatBasis(int dimension, int N, double alphaReal, double alphaImag) {
        this.dimension = dimension;
        this.N = N;
        this.alpha = new double[]{alphaReal, alphaImag};
        createBasis();
    }

    @Override
    public void createBasis() {
        StandardBasis fockStates = new StandardBasis(dimension);

        // Generate coherent state
        // Common factor
        double coherentFactor = Math.exp(-(Math.pow(alpha[0], 2) + Math.pow(alpha[1], 2)) / 2.0);
        long[] factorials = computeFactorials(N);
        double[] numerator;
        double denominator;
        double[] factor;

        ZMatrixRMaj coherentState = new ZMatrixRMaj(N, 1);
        for (int n = 0; n < N; n++) {
            numerator = complexPower(alpha, n);
            denominator = Math.sqrt(factorials[n]);
            factor = new double[]{numerator[0] / denominator, numerator[1] / denominator};

            ZMatrixRMaj temp = fockStates.getBasisStates()[n].copy();
            CommonOps_ZDRM.scale(factor[0], factor[1], temp);
            CommonOps_ZDRM.add(coherentState, temp, coherentState);
        }

        CommonOps_ZDRM.scale(coherentFactor, 1.0, coherentState);

        double[] phase;
        double[] phase_shift;
        basisStates = new ZMatrixRMaj[dimension];
        for (int ell = 0; ell < dimension; ell++) {
            phase_shift = new double[]{0.0, 0.0};
            basisStates[ell] = new ZMatrixRMaj(N, 1);
            for (int k = 0; k < N; k++) {
                phase = expI(2.0 * Math.PI * k / N * (N - ell));
                phase_shift[0] = phase_shift[0] + phase[0];
                phase_shift[1] = phase_shift[1] + phase[1];
            }

            basisStates[ell] = coherentState.copy();
            CommonOps_ZDRM.scale(phase_shift[0], phase_shift[1], basisStates[ell]);

            ZMatrixRMaj norm = new ZMatrixRMaj(1, 1);
            CommonOps_ZDRM.multTransA(basisStates[ell], basisStates[ell], norm);
            CommonOps_ZDRM.scale(1 / norm.getReal(0, 0), 1, basisStates[ell]);
        }

        createMatrix();
    }

    @Override
    public ZMatrixRMaj[] getBasisStates() {
        return basisStates;
    }

    @Override
    public ZMatrixRMaj getBasisMatrix() {
        return basisMatrix;
    }

    @Override
    public int getDimension() {
        return dimension; // the number of cat basis states
    }

    @Override
    public String toString() {
        return "Cat basis matrix of dimension " + getDimension() + ":\n" + MatrixIO.prettyFormat(getBasisMatrix());
    }

    private static long[] computeFactorials(int n) {
        long[] result = new long[n + 1];
        result[0] = 1;
        for (int i = 1; i <= n; i++) {
            result[i] = result[i - 1] * i;
        }
        return result;
    }

    private static double[] complexPower(double[] z, int n) {
        if (n == 0) {
            return new double[]{1.0, 0.0};
        }

        double r = Math.hypot(z[0], z[1]);
        double theta = Math.atan2(z[1], z[0]);

        double rPowered = Math.pow(r, n);
        double resultReal = rPowered * Math.cos(n * theta);
        double resultImag = rPowered * Math.sin(n * theta);

        return new double[]{resultReal, resultImag};
    }

    public static double[] expI(double theta) {
        return new double[]{Math.cos(theta), Math.sin(theta)};
    }

}