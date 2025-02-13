package quantum;

import quantum.io.Console;
import quantum.basis.StandardBasis;
import quantum.basis.CatBasis;

public class Quantum {
    private static Console console;

    public static void init() {
        console = Console.getInstance();
    }

    public Quantum() {
        init();
    }

    public static void main(String[] args) {
        new Quantum();
        console.write("Quantum simulation application started.");

        StandardBasis standardBasis = new StandardBasis(3);
        console.write(standardBasis.toString());
        console.write("");
        standardBasis.checkMatrix();

        CatBasis catBasis = new CatBasis(3, 3, 1000, 1000);
        console.write(catBasis.toString());
        console.write("");
        catBasis.checkMatrix();

        console.write("Quantum simulation application ended.");
    }
}