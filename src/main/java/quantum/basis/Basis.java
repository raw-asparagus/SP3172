package quantum.basis;

import org.ejml.data.ZMatrixRMaj;
import org.ejml.dense.row.CommonOps_ZDRM;

import quantum.io.MatrixIO;

public abstract class Basis {

    protected int dimension;

    protected ZMatrixRMaj[] basisStates;

    protected ZMatrixRMaj basisMatrix;

    protected abstract void createBasis();

    protected void createMatrix() {
        basisMatrix = new ZMatrixRMaj(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double real = basisStates[i].getReal(j, 0);
                double imag = basisStates[i].getImag(j, 0);
                basisMatrix.set(j, i, real, imag);
            }
        }
    }

    public void checkMatrix() {
        ZMatrixRMaj result = new ZMatrixRMaj(dimension, dimension);

        ZMatrixRMaj temp = new ZMatrixRMaj(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            CommonOps_ZDRM.multTransB(basisStates[i], basisStates[i], temp);
            CommonOps_ZDRM.add(result, temp, result);
        }

        System.out.println(MatrixIO.prettyFormat(result));
    }

    public abstract ZMatrixRMaj[] getBasisStates();

    public abstract ZMatrixRMaj getBasisMatrix();

    public abstract int getDimension();
}