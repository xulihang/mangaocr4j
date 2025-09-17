package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class MangaOCR {
    private OrtEnvironment env;
    private OrtSession session;
    private Map<Long, String> vocabMap; // 改用HashMap
    // 预分配缓冲区，避免重复创建
    private FloatBuffer imageBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224);

    // 添加时间统计变量
    private long preprocessTime = 0;
    private long generateTime = 0;
    private long decodeTime = 0;
    private long totalTime = 0;

    public MangaOCR(String modelPath, String vocabPath) throws Exception {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        // 优化会话选项
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

        session = env.createSession(modelPath, options);
        vocabMap = loadVocab(vocabPath);
    }

    public String run(Mat image) throws Exception {
        long startTime = System.nanoTime();

        float[][][][] inputImage = preprocess(image);
        List<Long> tokenIds = generate(inputImage);
        String text = decode(tokenIds);

        totalTime = System.nanoTime() - startTime;

        return text;
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

    // 添加获取时间统计的方法
    public Map<String, Double> getTimeStatistics() {
        Map<String, Double> stats = new HashMap<>();
        stats.put("preprocess", preprocessTime / 1_000_000.0);
        stats.put("generate", generateTime / 1_000_000.0);
        stats.put("decode", decodeTime / 1_000_000.0);
        stats.put("total", totalTime / 1_000_000.0);
        return stats;
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

        // 使用更高效的尺寸调整方法
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(224, 224), 0, 0, Imgproc.INTER_LINEAR);

        // 直接处理数据，避免多次转换
        Mat rgb = new Mat();
        int conversionCode = Imgproc.COLOR_BGR2RGB;
        if (resized.channels() == 1) {
            conversionCode = Imgproc.COLOR_GRAY2RGB;
        } else if (resized.channels() == 4) {
            conversionCode = Imgproc.COLOR_BGRA2RGB;
        }
        Imgproc.cvtColor(resized, rgb, conversionCode);

        // 一次性完成归一化和通道重排
        int height = rgb.rows();
        int width = rgb.cols();
        float[][][][] input = new float[1][3][height][width];

        byte[] data = new byte[(int) (rgb.total() * rgb.channels())];
        rgb.get(0, 0, data);

        // 直接处理字节数据，避免额外的浮点转换
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
        tokenIds.add(2L); // 开始符

        // 预填充图像张量（保持不变）
        imageBuffer.rewind();
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    imageBuffer.put(image[0][c][h][w]);
                }
            }
        }
        imageBuffer.rewind();

        // 重用图像张量
        try (OnnxTensor imageTensor = OnnxTensor.createTensor(env, imageBuffer, new long[]{1, 3, 224, 224})) {

            // 预分配token数组，避免重复创建
            long[] tokenArray = new long[100]; // 最大长度
            int currentLength = 0;

            // 添加重复检测和置信度阈值
            int consecutiveNonText = 0;
            final int MAX_CONSECUTIVE_NON_TEXT = 5;
            final float CONFIDENCE_THRESHOLD = 0.7f;
            final float TOO_LOW_CONFIDENCE_THRESHOLD = 0.2f;

            for (int step = 0; step < 100; step++) {
                // 更新token数组
                if (currentLength < tokenIds.size()) {
                    for (int i = 0; i < tokenIds.size(); i++) {
                        tokenArray[i] = tokenIds.get(i);
                    }
                    currentLength = tokenIds.size();
                }

                // 只使用实际长度的子数组
                long[] currentTokens = Arrays.copyOf(tokenArray, currentLength);

                try (OnnxTensor tokenTensor = OnnxTensor.createTensor(env, new long[][]{currentTokens});
                     OrtSession.Result results = session.run(Map.of(
                             "image", imageTensor,
                             "token_ids", tokenTensor
                     ))) {

                    float[][][] logits = (float[][][]) results.get(0).getValue();
                    int lastIndex = logits[0].length - 1;
                    float[] probabilities = softmax(logits[0][lastIndex]);
                    int tokenId = argmax(probabilities);
                    float confidence = probabilities[tokenId];

                    // 提前终止条件
                    if (tokenId == 3) break; // 结束符

                    // 低置信度或重复生成非文本token时提前终止
                    if (confidence < CONFIDENCE_THRESHOLD) {
                        if (confidence < TOO_LOW_CONFIDENCE_THRESHOLD) {
                            break;
                        }
                        consecutiveNonText++;
                        if (consecutiveNonText >= MAX_CONSECUTIVE_NON_TEXT) {
                            break;
                        }
                    } else {
                        consecutiveNonText = 0;
                    }

                    tokenIds.add((long) tokenId);
                }
            }
        }

        generateTime = System.nanoTime() - startTime;
        return tokenIds;
    }

    // 优化的softmax实现
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : logits) {
            if (value > max) max = value;
        }

        float sum = 0.0f;
        float[] expValues = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            // 使用快速指数近似
            expValues[i] = fastExp(logits[i] - max);
            sum += expValues[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < expValues.length; i++) {
            expValues[i] *= invSum;
        }
        return expValues;
    }

    // 快速指数函数近似
    private float fastExp(float x) {
        x = 1.0f + x / 256.0f;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }

    private String decode(List<Long> tokenIds) {
        long startTime = System.nanoTime();

        StringBuilder sb = new StringBuilder();
        for (long id : tokenIds) {
            if (id < 5) continue;
            String token = vocabMap.get(id);
            if (token != null) {
                sb.append(token);
            }
        }

        decodeTime = System.nanoTime() - startTime;
        return sb.toString();
    }

    private static float[] flatten(float[][][][] arr) {
        int d1 = arr.length, d2 = arr[0].length, d3 = arr[0][0].length, d4 = arr[0][0][0].length;
        float[] flat = new float[d1 * d2 * d3 * d4];
        int idx = 0;
        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++)
                for (int k = 0; k < d3; k++)
                    for (int l = 0; l < d4; l++)
                        flat[idx++] = arr[i][j][k][l];
        return flat;
    }

    // 使用循环展开优化argmax
    private static int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];

        // 循环展开以提高性能
        int i = 1;
        int length = arr.length;

        for (; i <= length - 4; i += 4) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
            if (arr[i + 1] > maxVal) {
                maxVal = arr[i + 1];
                maxIdx = i + 1;
            }
            if (arr[i + 2] > maxVal) {
                maxVal = arr[i + 2];
                maxIdx = i + 2;
            }
            if (arr[i + 3] > maxVal) {
                maxVal = arr[i + 3];
                maxIdx = i + 3;
            }
        }

        // 处理剩余元素
        for (; i < length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}