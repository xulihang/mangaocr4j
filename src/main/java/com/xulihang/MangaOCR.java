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
    public MangaOCR(String modelPath, String vocabPath) throws Exception {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
        vocabMap = loadVocab(vocabPath);
    }

    public String run(Mat image) throws Exception {
        float[][][][] inputImage = preprocess(image);
        List<Long> tokenIds = generate(inputImage);
        String text = decode(tokenIds);
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
        return input;
    }

    private List<Long> generate(float[][][][] image) throws Exception {
        List<Long> tokenIds = new ArrayList<>(50); // 预分配合理大小
        tokenIds.add(2L);

        // 预填充图像张量
        imageBuffer.rewind();
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    imageBuffer.put(image[0][c][h][w]);
                }
            }
        }
        imageBuffer.rewind();

        OnnxTensor imageTensor = OnnxTensor.createTensor(env, imageBuffer, new long[]{1, 3, 224, 224});

        try {
            for (int step = 0; step < 300; step++) {
                long[] tokenArray = new long[tokenIds.size()];
                for (int i = 0; i < tokenIds.size(); i++) {
                    tokenArray[i] = tokenIds.get(i);
                }

                OnnxTensor tokenTensor = OnnxTensor.createTensor(env, new long[][]{tokenArray});

                try (OrtSession.Result results = session.run(Map.of(
                        "image", imageTensor,
                        "token_ids", tokenTensor
                ))) {

                    float[][][] logits = (float[][][]) results.get(0).getValue();
                    int lastIndex = logits[0].length - 1;
                    int tokenId = argmax(logits[0][lastIndex]);

                    tokenIds.add((long) tokenId);
                    tokenTensor.close();

                    if (tokenId == 3) break;
                }
            }
        } finally {
            imageTensor.close();
        }

        return tokenIds;
    }

    private String decode(List<Long> tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (long id : tokenIds) {
            if (id < 5) continue;
            String token = vocabMap.get(id);
            if (token != null) {
                sb.append(token);
            }
        }
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