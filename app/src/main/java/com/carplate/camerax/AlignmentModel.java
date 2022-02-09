package com.carplate.camerax;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class AlignmentModel {
    private static final String MODEL_NAME = "alignment_v2.tflite";
    private static final int BATCH_SIZE = 1;
    private static final int IMG_HEIGHT = 128;
    private static final int IMG_WIDTH = 128;
    private static final int NUM_CHANNEL = 3;

    private final Interpreter.Options options = new Interpreter.Options();
    private final Interpreter mInterpreter;

    private final ByteBuffer mImageData;
    private final float[][] detectionResult = new float[1][8];
    private int[] mImagePixels = new int[IMG_HEIGHT * IMG_WIDTH];
    public AlignmentModel(Activity activity, Interpreter.Options options) throws IOException {
        mInterpreter = new Interpreter(loadModelFile(activity), options);
        mImageData = ByteBuffer.allocateDirect(
                4 * BATCH_SIZE * IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL);
        mImageData.order(ByteOrder.nativeOrder());

    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImageData == null) {
            return;
        }
        mImageData.rewind();
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.

        int pixel = 0;
        for (int i = 0; i < IMG_WIDTH; ++i) {
            for (int j = 0; j < IMG_HEIGHT; ++j) {
                final int val = mImagePixels[pixel++];
                int px = bitmap.getPixel(j,i);
                int r= Color.red(px);
                mImageData.putFloat(r/255.0f);
            }
        }
        pixel = 0;
        for (int i = 0; i < IMG_HEIGHT; ++i) {
            for (int j = 0; j < IMG_WIDTH; ++j) {
                final int val = mImagePixels[pixel++];
                int px = bitmap.getPixel(j,i);
                int g= Color.green(px);
                mImageData.putFloat(g/255.0f);
            }
        }
        pixel = 0;
        for (int i = 0; i < IMG_HEIGHT; ++i) {
            for (int j = 0; j < IMG_WIDTH; ++j) {
                final int val = mImagePixels[pixel++];
                int px = bitmap.getPixel(j,i);
                int b=Color.blue(px);
                mImageData.putFloat(b/255.0f);
            }
        }
    }

    public float[] getCoordinate(Bitmap bitmap){
        convertBitmapToByteBuffer(bitmap);
        mInterpreter.run(mImageData, detectionResult);
        return detectionResult[0];
    }
}
