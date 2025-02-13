package quantum.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.IOException;

public class Console {
    private static Console instance;
    private final BufferedReader reader;
    private final BufferedWriter writer;

    private Console() {
        reader = new BufferedReader(new InputStreamReader(System.in));
        writer = new BufferedWriter(new OutputStreamWriter(System.out));
    }

    public static synchronized Console getInstance() {
        if (instance == null) {
            instance = new Console();
        }
        return instance;
    }

    public String readLine() {
        try {
            return reader.readLine();
        } catch (IOException e) {
            error("Failed to read from console", e);
            return "";
        }
    }

    public void write(String message) {
        try {
            writer.write(message);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            error("Failed to write to console", e);
        }
    }

    public void error(String message, Exception e) {
        try {
            writer.write("[ERR] " + message + " - " + e.getMessage());
            writer.newLine();
            writer.flush();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }
}