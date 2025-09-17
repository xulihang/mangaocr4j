package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

public class MangaOCR {
    private OrtEnvironment env;
    private OrtSession encoderSession;
    private OrtSession decoderSession;
    private Map<Long, String> vocabMap;

    private FloatBuffer imageBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224);

    private long preprocessTime = 0;
    private long generateTime = 0;
    private long decodeTime = 0;
    private long totalTime = 0;

    public MangaOCR(String encoderPath, String decoderPath, String vocabPath) throws Exception {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

        encoderSession = env.createSession(encoderPath, options);
        decoderSession = env.createSession(decoderPath, options);
        vocabMap = loadVocab(vocabPath);
    }

    // 添加时间统计打印方法
    public void printTimeStatistics() {
        System.out.println("===== 时间统计 =====");
        System.out.printf("预处理时间: %.3f ms%n", preprocessTime / 1_000_000.0);
        System.out.printf("生成时间: %.3f ms%n", generateTime / 1_000_000.0);
        System.out.printf("解码时间: %.3f ms%n", decodeTime / 1_000_000.0);
        System.out.printf("总时间: %.3f ms%n", totalTime / 1_000_000.0);
        System.out.println("====================");
    }

    public String run(Mat image) throws Exception {
        long startTime = System.nanoTime();

        float[][][][] inputImage = preprocess(image);
        List<Long> tokenIds = generate(inputImage);
        String text = decode(tokenIds);

        totalTime = System.nanoTime() - startTime;
        return text;
    }

    private Map<Long, String> loadVocab(String vocabFile) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(vocabFile));
        Map<Long, String> map = new HashMap<>(lines.size());
        for (long i = 0; i < lines.size(); i++) {
            map.put(i, lines.get((int) i));
        }
        return map;
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

        imageBuffer.rewind();
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    imageBuffer.put(image[0][c][h][w]);
                }
            }
        }
        imageBuffer.rewind();

        // 1) Run encoder once
        OnnxTensor imageTensor = OnnxTensor.createTensor(env, imageBuffer, new long[]{1, 3, 224, 224});
        OrtSession.Result encoderResult = encoderSession.run(Map.of("pixel_values", imageTensor));
        Object encoderOutputs = encoderResult.get(0).getValue(); // usually float[1][seq][hidden]

        // 2) Iterative decoding
        long[] tokenArray = new long[100];
        int currentLength = 0;

        for (int step = 0; step < 100; step++) {
            if (currentLength < tokenIds.size()) {
                for (int i = 0; i < tokenIds.size(); i++) {
                    tokenArray[i] = tokenIds.get(i);
                }
                currentLength = tokenIds.size();
            }
            long[] currentTokens = Arrays.copyOf(tokenArray, currentLength);

            try (OnnxTensor tokenTensor = OnnxTensor.createTensor(env, new long[][]{currentTokens});
                 OnnxTensor encoderOutTensor = OnnxTensor.createTensor(env, encoderOutputs)) {

                OrtSession.Result results = decoderSession.run(Map.of(
                        "encoder_hidden_states", encoderOutTensor,
                        "input_ids", tokenTensor
                ));

                float[][][] logits = (float[][][]) results.get(0).getValue();
                int lastIndex = logits[0].length - 1;
                float[] probabilities = softmax(logits[0][lastIndex]);
                int tokenId = argmax(probabilities);
                float confidence = probabilities[tokenId];
                // 低置信度 → 提前结束
                if (confidence < 0.2f) {
                    break;
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

        float sum = 0.0f;
        float[] expValues = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = fastExp(logits[i] - max);
            sum += expValues[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < expValues.length; i++) expValues[i] *= invSum;
        return expValues;
    }

    private float fastExp(float x) {
        x = 1.0f + x / 256.0f;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        return x;
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
}
