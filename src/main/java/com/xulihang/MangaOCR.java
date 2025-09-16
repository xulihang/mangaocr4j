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
    private List<String> vocab;

    public MangaOCR(String modelPath, String vocabPath) throws Exception {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
        vocab = loadVocab(vocabPath);
    }

    public String run(Mat image) throws Exception {
        float[][][][] inputImage = preprocess(image);
        List<Long> tokenIds = generate(inputImage);
        String text = decode(tokenIds);
        return text;
    }

    private List<String> loadVocab(String vocabFile) throws IOException {
        return Files.readAllLines(Paths.get(vocabFile));
    }

    private float[][][][] preprocess(Mat image) {
        // resize to 224x224
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(224, 224));

        // convert to RGB if needed
        Mat rgb = new Mat();
        if (resized.channels() == 1) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
        } else if (resized.channels() == 4) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGRA2RGB);
        } else {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        }

        // float32 normalization
        rgb.convertTo(rgb, CvType.CV_32F, 1.0 / 255.0);

        int height = rgb.rows();
        int width = rgb.cols();
        int channels = rgb.channels();

        float[][][][] input = new float[1][3][height][width];

        float[] data = new float[(int) (rgb.total() * rgb.channels())];
        rgb.get(0, 0, data);

        // OpenCV stores as HWC, we need CHW
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    float val = data[(h * width + w) * channels + c];
                    val = (val - 0.5f) / 0.5f; // normalize
                    input[0][c][h][w] = val;
                }
            }
        }
        return input;
    }

    private List<Long> generate(float[][][][] image) throws Exception {
        List<Long> tokenIds = new ArrayList<>();
        tokenIds.add(2L); // BOS token

        for (int step = 0; step < 300; step++) {
            long[][] tokenArray = new long[1][tokenIds.size()];
            for (int i = 0; i < tokenIds.size(); i++) {
                tokenArray[0][i] = tokenIds.get(i);
            }

            OnnxTensor imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatten(image)), new long[]{1, 3, 224, 224});
            OnnxTensor tokenTensor = OnnxTensor.createTensor(env, tokenArray);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("image", imageTensor);
            inputs.put("token_ids", tokenTensor);

            OrtSession.Result results = session.run(inputs);
            float[][][] logits = (float[][][]) results.get(0).getValue();

            int lastIndex = logits[0].length - 1;
            float[] lastLogits = logits[0][lastIndex];

            int tokenId = argmax(lastLogits);
            tokenIds.add((long) tokenId);

            results.close();
            imageTensor.close();
            tokenTensor.close();

            if (tokenId == 3) { // EOS
                break;
            }
        }

        return tokenIds;
    }

    private String decode(List<Long> tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (long id : tokenIds) {
            if (id < 5) continue;
            if (id < vocab.size()) {
                sb.append(vocab.get((int) id));
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