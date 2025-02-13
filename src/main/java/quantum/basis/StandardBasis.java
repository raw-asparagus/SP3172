package quantum.basis;

import org.ejml.data.ZMatrixRMaj;

import quantum.io.MatrixIO;

public class StandardBasis extends Basis {

    public StandardBasis(int dimension) {
        this.dimension = dimension;
        createBasis();
    }

    @Override
    public void createBasis() {
        basisStates = new ZMatrixRMaj[dimension];
        for (int i = 0; i < dimension; i++) {
            ZMatrixRMaj state = new ZMatrixRMaj(dimension, 1);
            state.set(i, 0, 1.0, 0.0);
            basisStates[i] = state;
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
        return dimension;
    }

    @Override
    public String toString() {
        return "Standard basis matrix of dimension " + getDimension() + ":\n" + MatrixIO.prettyFormat(getBasisMatrix());
    }
}