package org.tensorflow.lite.examples.posenet.lib;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import androidx.annotation.RequiresApi;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import javax.xml.transform.Result;


public class TestMulti {
//    private final Registrar mRegistrar;
    private Interpreter tfLite;
    private boolean tfLiteBusy = false;
    private int inputSize = 0;
//    private Vector<String> labels;
    float[][] labelProb;
    private static final int BYTES_PER_CHANNEL = 4;

    String[] partNames = {
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    String[][] poseChain = {
            {"nose", "leftEye"}, {"leftEye", "leftEar"}, {"nose", "rightEye"},
            {"rightEye", "rightEar"}, {"nose", "leftShoulder"},
            {"leftShoulder", "leftElbow"}, {"leftElbow", "leftWrist"},
            {"leftShoulder", "leftHip"}, {"leftHip", "leftKnee"},
            {"leftKnee", "leftAnkle"}, {"nose", "rightShoulder"},
            {"rightShoulder", "rightElbow"}, {"rightElbow", "rightWrist"},
            {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
            {"rightKnee", "rightAnkle"}
    };

    Map<String, Integer> partsIds = new HashMap<>();
    List<Integer> parentToChildEdges = new ArrayList<>();
    List<Integer> childToParentEdges = new ArrayList<>();

    ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
        Tensor tensor = tfLite.getInputTensor(0);
        int[] shape = tensor.shape();
        inputSize = shape[1];
        int inputChannels = shape[3];

        int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
        imgData.order(ByteOrder.nativeOrder());

        Bitmap bitmap = bitmapRaw;
        if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
            Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                    inputSize, inputSize, false);
            bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
            final Canvas canvas = new Canvas(bitmap);
            canvas.drawBitmap(bitmapRaw, matrix, null);
        }

        if (tensor.dataType() == DataType.FLOAT32) {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
                    imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
                }
            }
        } else {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                }
            }
        }

        return imgData;
    }

    ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
        InputStream inputStream = new FileInputStream(path.replace("file://", ""));
        Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

        return feedInputTensor(bitmapRaw, mean, std);
    }

  private abstract class TfliteTask extends AsyncTask<Void, Void, Void> {
        Result result;
        boolean asynch;

        TfliteTask(HashMap args, Result result) {
            if (tfLiteBusy) throw new RuntimeException("Interpreter busy");
            else tfLiteBusy = true;
            Object asynch = args.get("asynch");
            this.asynch = asynch == null ? false : (boolean) asynch;
            this.result = result;
        }

        abstract void runTflite();

        abstract void onRunTfliteDone();

        public void executeTfliteTask() {
            if (asynch) execute();
            else {
                runTflite();
                tfLiteBusy = false;
                onRunTfliteDone();
            }
        }

        protected Void doInBackground(Void... backgroundArguments) {
            runTflite();
            return null;
        }

        protected void onPostExecute(Void backgroundResult) {
            tfLiteBusy = false;
            onRunTfliteDone();
        }
    }


    void runPoseNetOnImage(HashMap args, Result result) throws IOException {
        String path = args.get("path").toString();
        double mean = (double) (args.get("imageMean"));
        float IMAGE_MEAN = (float) mean;
        double std = (double) (args.get("imageStd"));
        float IMAGE_STD = (float) std;
        int numResults = (int) args.get("numResults");
        double threshold = (double) args.get("threshold");
        int nmsRadius = (int) args.get("nmsRadius");

        ByteBuffer imgData = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);

        new RunPoseNet(args, imgData, numResults, threshold, nmsRadius, result).executeTfliteTask();
    }


    void initPoseNet(Map<Integer, Object> outputMap) {
        if (partsIds.size() == 0) {
            for (int i = 0; i < partNames.length; ++i)
                partsIds.put(partNames[i], i);

            for (int i = 0; i < poseChain.length; ++i) {
                parentToChildEdges.add(partsIds.get(poseChain[i][1]));
                childToParentEdges.add(partsIds.get(poseChain[i][0]));
            }
        }

        for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
            int[] shape = tfLite.getOutputTensor(i).shape();
            float[][][][] output = new float[shape[0]][shape[1]][shape[2]][shape[3]];
            outputMap.put(i, output);
        }
    }

    private class RunPoseNet extends TfliteTask {
        long startTime;
        Object[] input;
        Map<Integer, Object> outputMap = new HashMap<>();
        int numResults;
        double threshold;
        int nmsRadius;

        int localMaximumRadius = 1;
        int outputStride = 16;

        RunPoseNet(HashMap args,
                   ByteBuffer imgData,
                   int numResults,
                   double threshold,
                   int nmsRadius,
                   Result result) throws IOException {
            super(args, result);
            this.numResults = numResults;
            this.threshold = threshold;
            this.nmsRadius = nmsRadius;

            input = new Object[]{imgData};
            initPoseNet(outputMap);

            startTime = SystemClock.uptimeMillis();
        }

        protected void runTflite() {
            tfLite.runForMultipleInputsOutputs(input, outputMap);
        }

        protected void onRunTfliteDone() {
            Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

            float[][][] scores = ((float[][][][]) outputMap.get(0))[0];
            float[][][] offsets = ((float[][][][]) outputMap.get(1))[0];
            float[][][] displacementsFwd = ((float[][][][]) outputMap.get(2))[0];
            float[][][] displacementsBwd = ((float[][][][]) outputMap.get(3))[0];

            PriorityQueue<Map<String, Object>> pq = buildPartWithScoreQueue(scores, threshold, localMaximumRadius);
            // object:{score, x,y, part_id}; sort by score;
            // local maximal scores;

            int numParts = scores[0][0].length;
            int numEdges = parentToChildEdges.size();
            int sqaredNmsRadius = nmsRadius * nmsRadius;

            List<Map<String, Object>> results = new ArrayList<>();

            while (results.size() < numResults && pq.size() > 0) {
                Map<String, Object> root = pq.poll();
                float[] rootPoint = getImageCoords(root, outputStride, numParts, offsets);

                if (withinNmsRadiusOfCorrespondingPoint(
                        results, sqaredNmsRadius, rootPoint[0], rootPoint[1], (int) root.get("partId")))
                    continue;

                Map<String, Object> keypoint = new HashMap<>();
                keypoint.put("score", root.get("score"));
                keypoint.put("part", partNames[(int) root.get("partId")]);
                keypoint.put("y", rootPoint[0] / inputSize);
                keypoint.put("x", rootPoint[1] / inputSize);
                //why divide by inputsize?

                Map<Integer, Map<String, Object>> keypoints = new HashMap<>();
                keypoints.put((int) root.get("partId"), keypoint);
                //arrange the key-points according to parts

                for (int edge = numEdges - 1; edge >= 0; --edge) {
                    int sourceKeypointId = parentToChildEdges.get(edge);
                    int targetKeypointId = childToParentEdges.get(edge);
                    // source --- son node;  target: farther node;  strange definition
                    if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                        keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
                                targetKeypointId, scores, offsets, outputStride, displacementsBwd);
                        keypoints.put(targetKeypointId, keypoint);
                    }
                }

                for (int edge = 0; edge < numEdges; ++edge) {
                    int sourceKeypointId = childToParentEdges.get(edge);
                    int targetKeypointId = parentToChildEdges.get(edge);
                    if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                        keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
                                targetKeypointId, scores, offsets, outputStride, displacementsFwd);
                        keypoints.put(targetKeypointId, keypoint);
                    }
                }

                Map<String, Object> result = new HashMap<>();
                result.put("keypoints", keypoints);
                result.put("score", getInstanceScore(keypoints, numParts));
                results.add(result);
            }

//            result.success(results);
        }
    }

    PriorityQueue<Map<String, Object>> buildPartWithScoreQueue(float[][][] scores,
                                                               double threshold,
                                                               int localMaximumRadius) {
        // choose the all the local maximal scores; and sort them by score;  TODO? why
        PriorityQueue<Map<String, Object>> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Map<String, Object>>() {
                            @Override
                            public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                                return Float.compare((float) rhs.get("score"), (float) lhs.get("score"));
                            }
                        });

        for (int heatmapY = 0; heatmapY < scores.length; ++heatmapY) {
            for (int heatmapX = 0; heatmapX < scores[0].length; ++heatmapX) {
                for (int keypointId = 0; keypointId < scores[0][0].length; ++keypointId) {
                    float score = sigmoid(scores[heatmapY][heatmapX][keypointId]);
                    if (score < threshold) continue;

                    if (scoreIsMaximumInLocalWindow(
                            keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        Map<String, Object> res = new HashMap<>();
                        res.put("score", score);
                        res.put("y", heatmapY);
                        res.put("x", heatmapX);
                        res.put("partId", keypointId);
                        pq.add(res);
                    }
                }
            }
        }

        return pq;
    }

    boolean scoreIsMaximumInLocalWindow(int keypointId,
                                        float score,
                                        int heatmapY,
                                        int heatmapX,
                                        int localMaximumRadius,
                                        float[][][] scores) {
        boolean localMaximum = true;
        int height = scores.length;
        int width = scores[0].length;

        int yStart = Math.max(heatmapY - localMaximumRadius, 0);
        int yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            int xStart = Math.max(heatmapX - localMaximumRadius, 0);
            int xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum) {
                break;
            }
        }

        return localMaximum;
    }

    float[] getImageCoords(Map<String, Object> keypoint,
                           int outputStride,
                           int numParts,
                           float[][][] offsets) {
        ////  map the coordinate from heatmap to original image
        int heatmapY = (int) keypoint.get("y");
        int heatmapX = (int) keypoint.get("x");
        int keypointId = (int) keypoint.get("partId");
        float offsetY = offsets[heatmapY][heatmapX][keypointId];
        float offsetX = offsets[heatmapY][heatmapX][keypointId + numParts];

        float y = heatmapY * outputStride + offsetY;
        float x = heatmapX * outputStride + offsetX;

        return new float[]{y, x};
    }

    boolean withinNmsRadiusOfCorrespondingPoint(List<Map<String, Object>> poses,
                                                float squaredNmsRadius,
                                                float y,
                                                float x,
                                                int keypointId) {
        for (Map<String, Object> pose : poses) {
            Map<Integer, Object> keypoints = (Map<Integer, Object>) pose.get("keypoints");
            Map<String, Object> correspondingKeypoint = (Map<String, Object>) keypoints.get(keypointId);
            float _x = (float) correspondingKeypoint.get("x") * inputSize - x;
            float _y = (float) correspondingKeypoint.get("y") * inputSize - y;
            float squaredDistance = _x * _x + _y * _y;
            if (squaredDistance <= squaredNmsRadius)
                return true;
        }

        return false;
    }

    Map<String, Object> traverseToTargetKeypoint(int edgeId,
                                                 Map<String, Object> sourceKeypoint,
                                                 int targetKeypointId,
                                                 float[][][] scores,
                                                 float[][][] offsets,
                                                 int outputStride,
                                                 float[][][] displacements) {
        int height = scores.length;
        int width = scores[0].length;
        int numKeypoints = scores[0][0].length;
        float sourceKeypointY = (float) sourceKeypoint.get("y") * inputSize;
        float sourceKeypointX = (float) sourceKeypoint.get("x") * inputSize;

        int[] sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX,
                outputStride, height, width);

        float[] displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);

        float[] displacedPoint = new float[]{
                sourceKeypointY + displacement[0],
                sourceKeypointX + displacement[1]
        };

        float[] targetKeypoint = displacedPoint;

        final int offsetRefineStep = 2;
        for (int i = 0; i < offsetRefineStep; i++) {
            int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
                    outputStride, height, width);

            int targetKeypointY = targetKeypointIndices[0];
            int targetKeypointX = targetKeypointIndices[1];

            float offsetY = offsets[targetKeypointY][targetKeypointX][targetKeypointId];
            float offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints];

            targetKeypoint = new float[]{
                    targetKeypointY * outputStride + offsetY,
                    targetKeypointX * outputStride + offsetX
            };
        }

        int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
                outputStride, height, width);

        float score = sigmoid(scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]);

        Map<String, Object> keypoint = new HashMap<>();
        keypoint.put("score", score);
        keypoint.put("part", partNames[targetKeypointId]);
        keypoint.put("y", targetKeypoint[0] / inputSize);
        keypoint.put("x", targetKeypoint[1] / inputSize);

        return keypoint;
    }

    int[] getStridedIndexNearPoint(float _y, float _x, int outputStride, int height, int width) {
        int y_ = Math.round(_y / outputStride);
        int x_ = Math.round(_x / outputStride);
        int y = y_ < 0 ? 0 : y_ > height - 1 ? height - 1 : y_;
        int x = x_ < 0 ? 0 : x_ > width - 1 ? width - 1 : x_;
        return new int[]{y, x};
    }

    float[] getDisplacement(int edgeId, int[] keypoint, float[][][] displacements) {
        int numEdges = displacements[0][0].length / 2;
        int y = keypoint[0];
        int x = keypoint[1];
        return new float[]{displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges]};
    }

    float getInstanceScore(Map<Integer, Map<String, Object>> keypoints, int numKeypoints) {
        float scores = 0;
        for (Map.Entry<Integer, Map<String, Object>> keypoint : keypoints.entrySet())
            scores += (float) keypoint.getValue().get("score");
        return scores / numKeypoints;
    }

    private float sigmoid(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    private static Matrix getTransformationMatrix(final int srcWidth,
                                                  final int srcHeight,
                                                  final int dstWidth,
                                                  final int dstHeight,
                                                  final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (srcWidth != dstWidth || srcHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) srcWidth;
            final float scaleFactorY = dstHeight / (float) srcHeight;

            if (maintainAspectRatio) {
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        matrix.invert(new Matrix());
        return matrix;
    }

    private void close() {
        if (tfLite != null)
            tfLite.close();
//        labels = null;
        labelProb = null;
    }
}