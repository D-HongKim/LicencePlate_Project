package com.carplate.camerax;



import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;



import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static android.Manifest.permission.CAMERA;

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {



    private int m_Camidx = 0;//front : 1, back : 0
    private CameraBridgeViewBase m_CameraView;

    private static final int CAMERA_PERMISSION_CODE = 200;


    private Button button; // 카메라 캡쳐버튼
    private Boolean activationButton; // 캡쳐버튼 활성 유무
    private ImageView imageView; // 시각화
    private Mat matInput;
    public TextView textView, tvTime; // textView: 번호판 tvTime: 추론시간

    public NnApiDelegate nnApiDelegate = null;
    GpuDelegate delegate = new GpuDelegate();

    public Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
    public Interpreter.Options options1 = (new Interpreter.Options()).setNumThreads(4);


    long start,end; // 전체 추론시간
    long[] inferenceTime = new long[3]; // 모델별 추론시간

    private Bitmap onFrame; // yolo input
    private Bitmap onFrame2; // alignment input
    private Bitmap onFrame3; // char input

    // 모델 정의
    DHDetectionModel detectionModel;
    AlignmentModel alignmentModel;
    CharModel charModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("Permission::","onCreate");

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        // 화면 구성요소
        m_CameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        activationButton = false;
        button = (Button) findViewById(R.id.button_capture);
        textView = (TextView) findViewById(R.id.textView);
        imageView = (ImageView) findViewById(R.id.imageView);
        tvTime = (TextView)findViewById(R.id.tvTime);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!activationButton) {
                    activationButton = true;
                    m_CameraView.enableView();
                    button.setText("중지");
                } else {
                    activationButton = false;
                    m_CameraView.disableView();
                    button.setText("시작");
                }
            }
        });

        m_CameraView.setVisibility(SurfaceView.VISIBLE);
        m_CameraView.setCvCameraViewListener(this);
        m_CameraView.setCameraIndex(m_Camidx);

        try{

            detectionModel = new DHDetectionModel(this, options);
            //Toast.makeText(this.getApplicationContext(), options.toString(), Toast.LENGTH_LONG).show();
            alignmentModel = new AlignmentModel(this, options1);
            charModel = new CharModel(this, options1);

        }
        catch(IOException e){
            e.printStackTrace();
        }

    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean _Permission = false; //변수 추가
        Log.d("Permission::","onStart");
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){//최소 버전보다 버전이 높은지 확인

            if(checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_CODE);
                _Permission = true;
            }
            else if(checkSelfPermission(CAMERA)== PackageManager.PERMISSION_GRANTED){
                _Permission = true;
            }
        }

        if(_Permission){
            //여기서 카메라뷰 받아옴
            Log.d("Permission::","Permission true, onCameraPermission으로 감");
            onCameraPermissionGranted();
        }
    }

    @Override
    public void onResume()
    {
        super.onResume();
        Log.d("Permission::","onResume");
        if (OpenCVLoader.initDebug()) {
            m_LoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
        Log.d("Permission::","onPause");
        if (m_CameraView != null)
            m_CameraView.disableView();
    }
    @Override
    public void onDestroy() {
        super.onDestroy();

        if (m_CameraView != null)
            m_CameraView.disableView();
    }

    private BaseLoaderCallback m_LoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    m_CameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    protected void onCameraPermissionGranted() {
        Log.d("Permission::","onCameraPermissionGranted 함수");
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(m_CameraView);
    }

    private void setFullScreen(){
        int screenWidth = getWindowManager().getDefaultDisplay().getWidth();

    }
    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @SuppressLint("LongLogTag")
    @Override

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        start = System.currentTimeMillis();
        matInput = inputFrame.rgba();
        Mat input = matInput.clone();
        Log.d("input log:: ","cols: "+input.cols()+" rows: "+input.rows());

        if(activationButton){

            Mat toDetImage = new Mat();
            Size sz = new Size(256, 192);
            Imgproc.resize(matInput, toDetImage, sz);
            onFrame = Bitmap.createBitmap(toDetImage.cols(), toDetImage.rows(), Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(toDetImage, onFrame);
            long yolo_s = System.currentTimeMillis();
            float[][] proposal = detectionModel.getProposal(onFrame, input);
            long yolo_e = System.currentTimeMillis();
            inferenceTime[0] = yolo_e-yolo_s;


            if(proposal[1][4] < 0.5){ // reject inference
                return matInput;
            }

            int w = matInput.width();
            int h = matInput.height();
            float[] coord = new float[8];
            float x_center= proposal[1][0];
            float y_center = proposal[1][1];
            float width = proposal[1][2];
            float height = proposal[1][3];

            coord[0]= (float) (x_center-0.5*width);
            coord[1]= (float) (y_center-0.5*height);
            coord[2]= (float) (x_center+0.5*width);
            coord[3]= (float) (y_center-0.5*height);
            coord[4]= (float) (x_center+0.5*width);
            coord[5]= (float) (y_center+0.5*height);
            coord[6]= (float) (x_center-0.5*width);
            coord[7]= (float) (y_center+0.5*height);

            int w_ = (int) (0.01 * w);
            int h_ = (int) (0.01 * h);

            int pt1_x = (int) ((w * coord[0] - w_) > 0 ? (w * coord[0] - w_) : (w * coord[0]));
            int pt1_y = (int) ((h * coord[1] - h_) > 0 ? (h * coord[1] - h_) : (h * coord[1]));


            int pt3_x = (int) ((w * coord[4] + w_) < w ? (w * coord[4] + w_) : (w * coord[4]));
            int pt3_y = (int) ((h * coord[5] + h_) < h ? (h * coord[5] + h_) : (h * coord[5]));

            int new_w = (int) (pt3_x-pt1_x);
            int new_h = (int) (pt3_y-pt1_y);

            Imgproc.rectangle(matInput, new Point(pt1_x,pt1_y), new Point(pt3_x,pt3_y),
                    new Scalar(0, 255, 0), 10);

            Log.d("log:: ","pt1_x: "+pt1_x+" pt1_y: "+pt1_y+" new_w: "+new_w+" new_h: "+new_h);

            Rect roi = new Rect(pt1_x, pt1_y, new_w, new_h);
            Log.d("log:: ","x: "+roi.x+" y: "+roi.y+" w: "+roi.width+" h: "+roi.height);
            Log.d("input log:: ","cols: "+input.cols()+" rows: "+input.rows());
            if(roi.x + roi.width>input.cols() || roi.x<0 || roi.width<0 || roi.y+roi.height>input.rows() || roi.y<0 || roi.height<0)
                return matInput;
            Mat croppedImage = new Mat(input, roi);
            Mat toDetImage2 = new Mat();
            Size sz2 = new Size(128, 128);
            Imgproc.resize(croppedImage, toDetImage2, sz2);
            onFrame2 = Bitmap.createBitmap(toDetImage2.cols(), toDetImage2.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(toDetImage2, onFrame2);

            long align_s = System.currentTimeMillis();
            float[] coord2 = alignmentModel.getCoordinate(onFrame2);
            long align_e = System.currentTimeMillis();
            inferenceTime[1] = align_e-align_s;
            float[] new_coord2 = new float[8];

            // sigmoid
            for(int i=0;i<8;i++){
                new_coord2[i] = (float) (Math.exp(-(coord2[i]))+1);
                new_coord2[i] = 1/new_coord2[i];
                new_coord2[i] = new_coord2[i]*128;
            }

            // perspective transformation
            Mat outputImage = new Mat(256, 128, CvType.CV_8UC3);
            List<Point> src_pnt = new ArrayList<Point>();

            Point p0 = new Point(new_coord2[0], new_coord2[1]);        Point p1 = new Point(new_coord2[2], new_coord2[3]);
            Point p2 = new Point(new_coord2[4], new_coord2[5]);        Point p3 = new Point(new_coord2[6], new_coord2[7]);

            src_pnt.add(p0);
            src_pnt.add(p1);
            src_pnt.add(p2);
            src_pnt.add(p3);

            Mat startM = Converters.vector_Point2f_to_Mat(src_pnt);
            List<Point> dst_pnt = new ArrayList<Point>();
            Point p4 = new Point(0, 0);
            Point p5 = new Point(255, 0);
            Point p6 = new Point(255, 127);
            Point p7 = new Point(0, 127);
            dst_pnt.add(p4);
            dst_pnt.add(p5);
            dst_pnt.add(p6);
            dst_pnt.add(p7);
            Mat endM = Converters.vector_Point2f_to_Mat(dst_pnt);
            Mat M = Imgproc.getPerspectiveTransform(startM, endM);
            Size size2 = new Size(256, 128);
            Imgproc.warpPerspective(toDetImage2, outputImage, M, size2, Imgproc.INTER_CUBIC+ Imgproc.CV_WARP_FILL_OUTLIERS);
            onFrame3 = Bitmap.createBitmap(outputImage.cols(), outputImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(outputImage, onFrame3);

            // char prediction
            long char_s = System.currentTimeMillis();
            String result = charModel.getString(onFrame3);
            String result2 = result.substring(0,3) + " " + result.substring(3);
            long char_e = System.currentTimeMillis();
            inferenceTime[2] = char_e-char_s;
            end = System.currentTimeMillis();
            double fps = Math.round(((1.0/(end-start))*1000*100.0))/100.0;
            String infer_result = fps + "  fps";

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        tvTime.setText(infer_result);
                        textView.setText(result2);
                        imageView.setImageBitmap(onFrame3);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }
        else{

            m_CameraView.enableView();
        }
        return matInput;
    }
}