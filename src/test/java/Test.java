import com.xulihang.MangaOCR;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Test {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MangaOCR ocr = new MangaOCR("ocr/encoder_int8.onnx", "ocr/decoder_int8.onnx","ocr/vocab.txt");
        Mat img = Imgcodecs.imread("image1.jpg");
        for (int i = 0; i < 150; i++) {
            String text = ocr.run(img);
            ocr.printTimeStatistics();
            System.out.println(text);
        }
        String text2 = ocr.run(img);
        ocr.printTimeStatistics();
        System.out.println(text2);
        System.out.println("Hello world!");
    }
}
