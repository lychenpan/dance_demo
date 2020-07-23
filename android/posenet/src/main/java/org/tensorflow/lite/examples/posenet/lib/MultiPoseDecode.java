package org.tensorflow.lite.examples.posenet.lib;

import android.os.SystemClock;
import android.util.Log;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.examples.posenet.lib.Person;
import org.tensorflow.lite.examples.posenet.lib.BodyPart;
import org.tensorflow.lite.examples.posenet.lib.KeyPoint;
import org.tensorflow.lite.examples.posenet.lib.Position;


public class MultiPoseDecode {

    private float threshold = 0.7f;
    private int inputSize = 257; //model's input size; hard code
    private int nmsRadius = 10;
    private int localMaximumRadius = 1;
    private int numResults = 2;
    private int outputStride = 16;
/**
 * examples:
 * var result = await runPoseNetOnImage(
 *   path: filepath,     // required
 *   imageMean: 125.0,   // defaults to 125.0
 *   imageStd: 125.0,    // defaults to 125.0
 *   numResults: 2,      // defaults to 5
 *   threshold: 0.7,     // defaults to 0.5
 *   nmsRadius: 10,      // defaults to 20
 *   asynch: true        // defaults to true
 * );
 *
 * **/

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

    public MultiPoseDecode() {
        if (partsIds.size() == 0) {
            for (int i = 0; i < partNames.length; ++i)
                partsIds.put(partNames[i], i);

            for (int i = 0; i < poseChain.length; ++i) {
                parentToChildEdges.add(partsIds.get(poseChain[i][1]));
                childToParentEdges.add(partsIds.get(poseChain[i][0]));
            }
        }

    }

    public List<Map<String, Object>> decodeOutputMap(Map<Integer, Object> outputMap) {
//        Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));

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

        return results;
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

    private boolean scoreIsMaximumInLocalWindow(int keypointId,
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
//            if (correspondingKeypoint==null || !correspondingKeypoint.containsKey("x")){
//                return false;
//            }
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


}
