import com.xulihang.MangaOCR;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Test {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        MangaOCR ocr = new MangaOCR("ocr/manga-ocr.converted.encoder.preprocessed.quant.onnx", "ocr/manga-ocr.converted.decoder.preprocessed.quant.onnx","ocr/vocab.txt");
        Mat img = Imgcodecs.imread("image.jpg");
        String text = ocr.run(img);
        ocr.printTimeStatistics();
        System.out.println(text);
        String text2 = ocr.run(img);
        ocr.printTimeStatistics();
        System.out.println(text2);
        System.out.println("Hello world!");
    }
}
