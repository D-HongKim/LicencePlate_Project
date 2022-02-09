package com.carplate.camerax;


import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class DHDetectionModel{
    private static final String MODEL_NAME = "yolov5_0901-fp16.tflite";
    private static final int BATCH_SIZE = 1;
    public static final int IMG_HEIGHT = 192;
    public static final int IMG_WIDTH = 256;
    private static final int NUM_CHANNEL = 3;

    private final Interpreter.Options options = new Interpreter.Options();
    private final Interpreter mInterpreter;

    private final ByteBuffer mImageData;
    private final float[][][] detectionResult = new float[1][3024][6];
    private int[] mImagePixels = new int[IMG_HEIGHT * IMG_WIDTH]; // ??

    public DHDetectionModel(Activity activity, Interpreter.Options options) throws IOException{
        mInterpreter = new Interpreter(loadModelFile(activity), options);
        mImageData = ByteBuffer.allocateDirect(
                4 * BATCH_SIZE * IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL);
        mImageData.order(ByteOrder.nativeOrder());
        Tensor t = mInterpreter.getOutputTensor(mInterpreter.getOutputIndex("Identity"));

    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer mp = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        return mp;
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImageData == null) {
            return;
        }

        mImageData.rewind();
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < IMG_HEIGHT; ++i) {
            for (int j = 0; j < IMG_WIDTH; ++j) {
                final int val = mImagePixels[pixel++];
                int px = bitmap.getPixel(j,i);
                int r=Color.red(px);
                int g= Color.green(px);
                int b=Color.blue(px);

                mImageData.putFloat(r/255.0f);
                mImageData.putFloat(g/255.0f);
                mImageData.putFloat(b/255.0f);
            }
        }
        long endTime = SystemClock.uptimeMillis();
    }

    public float[][] getProposal(Bitmap bm, Mat input){

        float[][] ret = new float[2][5];

        convertBitmapToByteBuffer(bm);
        ret[0][0] = 0;
        mInterpreter.run(mImageData, detectionResult); //추론 detectionResult shape -> 1, 18900, 6
        ret[0][1] = 0;


        float max_conf = detectionResult[0][0][4];
        int idx = 0;
        for(int i = 0; i<3024; i++){
            if(max_conf < detectionResult[0][i][4]){
                max_conf = detectionResult[0][i][4];
                idx = i;

            }
        }
        detectionResult[0][idx][0] *= 256;
        detectionResult[0][idx][1] *= 192;
        detectionResult[0][idx][2] *= 256;
        detectionResult[0][idx][3] *= 192;
        float x1 = detectionResult[0][idx][0] - detectionResult[0][idx][2]/2;
        float y1 = detectionResult[0][idx][1] - detectionResult[0][idx][3]/2;
        float x2 = detectionResult[0][idx][0] + detectionResult[0][idx][2]/2;
        float y2 = detectionResult[0][idx][1] + detectionResult[0][idx][3]/2;
        ret[0][2] =0;
        ret[0][3] =0;
        ret[0][4] =0;
        ret[1][4] = max_conf;
        float gain_w = (float) (256.0 / input.cols());
        float gain_h = (float) (192.0 / input.rows());
        float [] pad = new float[2];

        pad[0]=0;
        pad[1]=0;
        float a =  ((x1-pad[0]) / gain_w );
        float b = ((y1-pad[1] )/ gain_h);
        float c = ((x2-pad[0]) / gain_w);
        float d = ((y2-pad[1]) / gain_h);
        float x_c = (a+c)/2;
        float y_c = (b+d)/2;
        float w = c-a;
        float h = d-b;

        ret[1][0] = x_c/input.cols();
        ret[1][1] = y_c/input.rows();
        ret[1][2] = w/input.cols();
        ret[1][3] = h/input.rows();
        return ret;
    }

}

