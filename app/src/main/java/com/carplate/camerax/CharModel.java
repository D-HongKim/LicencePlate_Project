package com.carplate.camerax;

import android.annotation.SuppressLint;
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


public class CharModel {
    private static final String MODEL_NAME = "char_v3.tflite";
    private static final int BATCH_SIZE = 1;
    private static final int IMG_HEIGHT = 128;
    private static final int IMG_WIDTH = 256;
    private static final int NUM_CHANNEL = 3;

    private final Interpreter mInterpreter;
    public Interpreter.Options options = (new Interpreter.Options());
    private final ByteBuffer mImageData;
    private final float[][][][] charResult = new float[1][7][45][1]; //?? input : 1 3 128 256 -> output :
    private int[] mImagePixels = new int[IMG_HEIGHT * IMG_WIDTH];

    static final String[] charTable = {"가","나","다","라","마","거","너","더","러","머","버","서","어","저","고",
    "노","도","로","모","보","소","오","조","구","누","두","루","무","부","수","우","주","허","하","호","0","1","2","3","4","5","6","7","8","9"};

    public CharModel(Activity activity, Interpreter.Options options) throws IOException {
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
        for (int i = 0; i < IMG_HEIGHT; ++i) {
            for (int j = 0; j < IMG_WIDTH; ++j) {
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

    @SuppressLint("LongLogTag")
    public String getString(Bitmap bm){
        String pred_str = "";
        convertBitmapToByteBuffer(bm);
        mInterpreter.run(mImageData, charResult); // charResult -> 1, 7, 45, 1
        float max;
        int idx = 0;
        for(int i=0;i<7; i++){
            max = -10;
            switch(i){
                case 2:
                    for(int j=0;j<35;j++){
                        if(max < charResult[0][i][j][0]){
                            max = charResult[0][i][j][0];
                            idx=j;
                        }
                    }
                    pred_str += charTable[idx];
                    break;
                default:
                    for(int j=35;j<45;j++){
                        if(max < charResult[0][i][j][0]){
                            max = charResult[0][i][j][0];
                            idx=j;
                        }
                    }
                    pred_str += charTable[idx];
                    break;
            }
        }
        return pred_str;
    }
}
