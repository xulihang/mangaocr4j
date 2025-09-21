package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class MangaOCR implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession encoderSession;
    private final OrtSession decoderSession;
    private final Map<Long, String> vocabMap;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private long preprocessTime = 0;
    private long generateTime = 0;
    private long decodeTime = 0;
    private long totalTime = 0;

    public MangaOCR(String encoderPath, String decoderPath, List<String> vocabLines) throws Exception {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);
        vocabMap = loadVocab(vocabLines);
    }

    public MangaOCR(byte[] encoder, byte[] decoder, List<String> vocabLines) throws Exception {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

        encoderSession = env.createSession(encoder, options);
        decoderSession = env.createSession(decoder, options);
        vocabMap = loadVocab(vocabLines);
    }

    // ---- Public API ----
    public String run(Mat image) throws Exception {
        if (closed.get()) throw new IllegalStateException("MangaOCR is closed");
        long startTime = System.nanoTime();

        float[][][][] inputImage = preprocess(image); // local array
        List<Long> tokenIds = generate(inputImage);   // performs encoder+decoder with proper closing
        String text = decode(tokenIds);

        totalTime = System.nanoTime() - startTime;
        return text;
    }

    public void printTimeStatistics() {
        System.out.println("===== 时间统计 =====");
        System.out.printf("预处理时间: %.3f ms%n", preprocessTime / 1_000_000.0);
        System.out.printf("生成时间: %.3f ms%n", generateTime / 1_000_000.0);
        System.out.printf("解码时间: %.3f ms%n", decodeTime / 1_000_000.0);
        System.out.printf("总时间: %.3f ms%n", totalTime / 1_000_000.0);
        System.out.println("====================");
    }

    @Override
    public void close() {
        if (!closed.getAndSet(true)) {
            try {
                encoderSession.close();
            } catch (Exception ignored) {}
            try {
                decoderSession.close();
            } catch (Exception ignored) {}
            // Don't close env; OrtEnvironment is usually shared. If you created a dedicated env, close it here.
            // env.close(); // only if you want to close environment
        }
    }

    // ---- Private helpers ----
    private Map<Long, String> loadVocab(List<String> lines) throws IOException {
        Map<Long, String> map = new HashMap<>(lines.size());
        for (long i = 0; i < lines.size(); i++) {
            map.put(i, lines.get((int) i));
        }
        return Collections.unmodifiableMap(map);
    }

    private float[][][][] preprocess(Mat image) {
        long startTime = System.nanoTime();

        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(224, 224), 0, 0, Imgproc.INTER_LINEAR);

        Mat rgb = new Mat();
        int conversionCode = Imgproc.COLOR_BGR2RGB;
        if (resized.channels() == 1) conversionCode = Imgproc.COLOR_GRAY2RGB;
        else if (resized.channels() == 4) conversionCode = Imgproc.COLOR_BGRA2RGB;
        Imgproc.cvtColor(resized, rgb, conversionCode);

        int height = rgb.rows();
        int width = rgb.cols();
        float[][][][] input = new float[1][3][height][width];

        byte[] data = new byte[(int) (rgb.total() * rgb.channels())];
        rgb.get(0, 0, data);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = (h * width + w) * 3;
                for (int c = 0; c < 3; c++) {
                    float val = (data[idx + c] & 0xFF) / 255.0f;
                    input[0][c][h][w] = (val - 0.5f) / 0.5f;
                }
            }
        }

        resized.release();
        rgb.release();

        preprocessTime = System.nanoTime() - startTime;
        return input;
    }

    private List<Long> generate(float[][][][] image) throws Exception {
        long startTime = System.nanoTime();

        List<Long> tokenIds = new ArrayList<>(50);
        tokenIds.add(2L); // start token

        // create a direct FloatBuffer local to this invocation (thread-local semantics)
        final int elems = 1 * 3 * 224 * 224;
        FloatBuffer imageBuffer = ByteBuffer
                .allocateDirect(elems * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        // fill the buffer
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    imageBuffer.put(image[0][c][h][w]);
                }
            }
        }
        imageBuffer.rewind();

        // 1) Run encoder once and make a deep copy of outputs (so Ort native memory can be freed)
        float[][][] encoderOutputsCopy; // shape expected [1][seq][hidden] or similar

        try (OnnxTensor imageTensor = OnnxTensor.createTensor(env, imageBuffer, new long[]{1, 3, 224, 224});
             OrtSession.Result encoderResult = encoderSession.run(Collections.singletonMap("pixel_values", imageTensor))) {

            // get raw value (usually float[][][])
            Object rawEnc = encoderResult.get(0).getValue();
            if (!(rawEnc instanceof float[][][])) {
                // defensive: try to handle other shapes, but we expect float[][][]
                throw new IllegalStateException("Unexpected encoder output type: " + rawEnc.getClass());
            }
            float[][][] enc = (float[][][]) rawEnc;

            // deep copy enc -> encoderOutputsCopy to avoid holding references to OrtValue
            int dim0 = enc.length;
            int dim1 = enc[0].length;
            int dim2 = enc[0][0].length;
            encoderOutputsCopy = new float[dim0][dim1][dim2];
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    System.arraycopy(enc[i][j], 0, encoderOutputsCopy[i][j], 0, dim2);
                }
            }
        }

        // 2) Iterative decoding
        long[] tokenArray = new long[100];
        int currentLength = 0;
        int consecutive_low_conf_times = 0;
        for (int step = 0; step < 100; step++) {
            if (currentLength < tokenIds.size()) {
                for (int i = 0; i < tokenIds.size(); i++) {
                    tokenArray[i] = tokenIds.get(i);
                }
                currentLength = tokenIds.size();
            }
            long[] currentTokens = Arrays.copyOf(tokenArray, currentLength);

            // tokenTensor: shape [1, seq]
            long[][] token2d = new long[1][currentTokens.length];
            System.arraycopy(currentTokens, 0, token2d[0], 0, currentTokens.length);

            try (OnnxTensor tokenTensor = OnnxTensor.createTensor(env, token2d);
                 OnnxTensor encoderOutTensor = OnnxTensor.createTensor(env, encoderOutputsCopy);
                 OrtSession.Result results = decoderSession.run(Map.of(
                         "encoder_hidden_states", encoderOutTensor,
                         "input_ids", tokenTensor
                 ))) {

                Object rawLogits = results.get(0).getValue();
                if (!(rawLogits instanceof float[][][])) {
                    throw new IllegalStateException("Unexpected decoder logits type: " + rawLogits.getClass());
                }
                float[][][] logits = (float[][][]) rawLogits;
                int lastIndex = logits[0].length - 1;
                float[] probabilities = softmax(logits[0][lastIndex]);
                int tokenId = argmax(probabilities);
                float confidence = probabilities[tokenId];
                if (confidence < 0.2f) {
                    consecutive_low_conf_times++;
                    if (consecutive_low_conf_times > 5) {
                        break;
                    }
                }else{
                    consecutive_low_conf_times = 0;
                }
                if (tokenId == 3) break; // end token
                tokenIds.add((long) tokenId);
            }
        }

        generateTime = System.nanoTime() - startTime;
        return tokenIds;
    }

    private String decode(List<Long> tokenIds) {
        long startTime = System.nanoTime();

        StringBuilder sb = new StringBuilder();
        for (long id : tokenIds) {
            if (id < 5) continue;
            String token = vocabMap.get(id);
            if (token != null) sb.append(token);
        }

        decodeTime = System.nanoTime() - startTime;
        return sb.toString();
    }

    // ---- Utility functions ----
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : logits) if (value > max) max = value;

        double sum = 0.0;
        double[] expValues = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - max);
            sum += expValues[i];
        }

        float invSum = (float) (1.0 / sum);
        float[] out = new float[logits.length];
        for (int i = 0; i < logits.length; i++) out[i] = (float) expValues[i] * invSum;
        return out;
    }

    private static int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Optional: expose timing fields or getters if needed
}