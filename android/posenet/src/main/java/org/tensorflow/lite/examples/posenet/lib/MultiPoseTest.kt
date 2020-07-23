package org.tensorflow.lite.examples.posenet.lib

import java.util.*
import kotlin.collections.HashMap


class MultiPoseTest {


    private val threshold = 0.7f
    private val inputSize = 257 //model's input size; hard code

    private val nmsRadius = 10
    private val localMaximumRadius = 1
    private val numResults = 2
    private val outputStride = 16

    /**
     * examples:
     * var result = await runPoseNetOnImage(
     * path: filepath,     // required
     * imageMean: 125.0,   // defaults to 125.0
     * imageStd: 125.0,    // defaults to 125.0
     * numResults: 2,      // defaults to 5
     * threshold: 0.7,     // defaults to 0.5
     * nmsRadius: 10,      // defaults to 20
     * asynch: true        // defaults to true
     * );
     *
     */
    var partNames = arrayOf(
        "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
        "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
        "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    )

    var poseChain = arrayOf(
        arrayOf("nose", "leftEye"),
        arrayOf("leftEye", "leftEar"),
        arrayOf("nose", "rightEye"),
        arrayOf("rightEye", "rightEar"),
        arrayOf("nose", "leftShoulder"),
        arrayOf("leftShoulder", "leftElbow"),
        arrayOf("leftElbow", "leftWrist"),
        arrayOf("leftShoulder", "leftHip"),
        arrayOf("leftHip", "leftKnee"),
        arrayOf("leftKnee", "leftAnkle"),
        arrayOf("nose", "rightShoulder"),
        arrayOf("rightShoulder", "rightElbow"),
        arrayOf("rightElbow", "rightWrist"),
        arrayOf("rightShoulder", "rightHip"),
        arrayOf("rightHip", "rightKnee"),
        arrayOf("rightKnee", "rightAnkle")
    )

    var partsIds: MutableMap<String, Int> = HashMap();
    var parentToChildEdges: MutableList<Int> = ArrayList()
    var childToParentEdges: MutableList<Int> = ArrayList()

    fun init(){
        if (partsIds.size == 0) {
            for (i in org.tensorflow.lite.examples.posenet.lib.partNames.indices) {
                partsIds.put(partNames[i],i)
            }
            for (i in poseChain.indices) {
                parentToChildEdges.add(partsIds[poseChain[i][1]]!!.toInt())
                childToParentEdges.add(partsIds[poseChain[i][0]]!!.toInt())
            }
        }
    }


    fun decodeOutputMap(outputMap: Map<Int?, Any?>): List<Map<String, Any>>? {
//        Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
        val scores =
            (outputMap[0] as Array<Array<Array<FloatArray>>>?)!![0]
        val offsets =
            (outputMap[1] as Array<Array<Array<FloatArray>>>?)!![0]
        val displacementsFwd =
            (outputMap[2] as Array<Array<Array<FloatArray>>>?)!![0]
        val displacementsBwd =
            (outputMap[3] as Array<Array<Array<FloatArray>>>?)!![0]
        val pq =
            buildPartWithScoreQueue(scores, threshold.toDouble(), localMaximumRadius)
        // object:{score, x,y, part_id}; sort by score;
        // local maximal scores;
        val numParts: Int = scores[0][0].size
        val numEdges = parentToChildEdges.size
        val sqaredNmsRadius = nmsRadius * nmsRadius
        val results: MutableList<Map<String, Any>> =
            ArrayList()
        while (results.size < numResults && pq.size > 0) {
            val root = pq.poll()
            val rootPoint = getImageCoords(root, outputStride, numParts, offsets)
            if (withinNmsRadiusOfCorrespondingPoint(
                    results,
                    sqaredNmsRadius.toFloat(),
                    rootPoint[0],
                    rootPoint[1],
                    root["partId"] as Int
                )
            ) continue
            var keypoint: MutableMap<String?, Any?> =
                HashMap()
            keypoint["score"] = root["score"]
            keypoint["part"] = partNames[root["partId"] as Int]
            keypoint["y"] = rootPoint[0] / inputSize
            keypoint["x"] = rootPoint[1] / inputSize
            //why divide by inputsize?
            val keypoints: MutableMap<Int, Map<String?, Any?>?> =
                HashMap()
            keypoints[root["partId"] as Int] = keypoint
            //arrange the key-points according to parts
            for (edge in numEdges - 1 downTo 0) {
                val sourceKeypointId = parentToChildEdges[edge]
                val targetKeypointId = childToParentEdges[edge]
                // source --- son node;  target: farther node;  strange definition
                if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(
                        targetKeypointId
                    )
                ) {
                    keypoint = traverseToTargetKeypoint(
                        edge, keypoints[sourceKeypointId],
                        targetKeypointId, scores, offsets, outputStride, displacementsBwd
                    )
                    keypoints[targetKeypointId] = keypoint
                }
            }
            for (edge in 0 until numEdges) {
                val sourceKeypointId = childToParentEdges[edge]
                val targetKeypointId = parentToChildEdges[edge]
                if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(
                        targetKeypointId
                    )
                ) {
                    keypoint = traverseToTargetKeypoint(
                        edge, keypoints[sourceKeypointId],
                        targetKeypointId, scores, offsets, outputStride, displacementsFwd
                    )
                    keypoints[targetKeypointId] = keypoint
                }
            }
            val result: MutableMap<String, Any> =
                HashMap()

            //TODO:  change format;

            //class KeyPoint {
            //  var bodyPart: BodyPart = BodyPart.NOSE
            //  var position: Position = Position()
            //  var score: Float = 0.0f
            //}
            //class Position {
            //  var x: Int = 0
            //  var y: Int = 0
            //}

//            val keypointList = Array(17) { KeyPoint() }
//            for(i in 0 until 16){
//                val tkp = KeyPoint()
//                tkp.bodyPart = BodyPart.values()[i]
//                tkp.position = Position()
//                keypointList.set(i, KeyPoint())
//            }
//            var totalScore = 0.0f
//            enumValues<BodyPart>().forEachIndexed { idx, it ->
//                keypointList[idx].bodyPart = it
//                keypointList[idx].position.x = xCoords[idx]
//                keypointList[idx].position.y = yCoords[idx]
//                keypointList[idx].score = confidenceScores[idx]
//                totalScore += confidenceScores[idx]
//            }
//
//            person.keyPoints = keypointList.toList()
//            person.score = totalScore / numKeypoints


            result["keypoints"] = keypoints
            result["score"] = getInstanceScore(keypoints, numParts)
            results.add(result)
        }
        return results
    }


    class pqComparator : Comparator<Map<String,Any>> {
        override fun compare(p1: Map<String, Any>, p2: Map<String, Any>): Int {
            return java.lang.Float.compare(
                p2.get("score") as Float,
                p1.get("score") as Float
            )
        }
    }

    fun buildPartWithScoreQueue(
        scores: Array<Array<FloatArray>>,
        threshold: Double,
        localMaximumRadius: Int
    ): PriorityQueue<Map<String, Any>> {
        // choose the all the local maximal scores; and sort them by score;  TODO? why
        val pq = PriorityQueue<Map<String, Any>>(1, pqComparator())
        for (heatmapY in scores.indices) {
            for (heatmapX in 0 until scores[0].size) {
                for (keypointId in 0 until scores[0][0].size) {
                    val score =
                        sigmoid(scores[heatmapY][heatmapX][keypointId])
                    if (score < threshold) continue
                    if (scoreIsMaximumInLocalWindow(
                            keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores
                        )
                    ) {
                        val res: MutableMap<String, Any> =
                            HashMap()
                        res["score"] = score
                        res["y"] = heatmapY
                        res["x"] = heatmapX
                        res["partId"] = keypointId
                        pq.add(res)
                    }
                }
            }
        }
        return pq
    }

    private fun scoreIsMaximumInLocalWindow(
        keypointId: Int,
        score: Float,
        heatmapY: Int,
        heatmapX: Int,
        localMaximumRadius: Int,
        scores: Array<Array<FloatArray>>
    ): Boolean {
        var localMaximum = true
        val height = scores.size
        val width: Int = scores[0].size
        val yStart = Math.max(heatmapY - localMaximumRadius, 0)
        val yEnd = Math.min(heatmapY + localMaximumRadius + 1, height)
        for (yCurrent in yStart until yEnd) {
            val xStart = Math.max(heatmapX - localMaximumRadius, 0)
            val xEnd = Math.min(heatmapX + localMaximumRadius + 1, width)
            for (xCurrent in xStart until xEnd) {
                if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
                    localMaximum = false
                    break
                }
            }
            if (!localMaximum) {
                break
            }
        }
        return localMaximum
    }

    fun getImageCoords(
        keypoint: Map<String, Any>,
        outputStride: Int,
        numParts: Int,
        offsets: Array<Array<FloatArray>>
    ): FloatArray {
        ////  map the coordinate from heatmap to original image
        val heatmapY = keypoint["y"] as Int
        val heatmapX = keypoint["x"] as Int
        val keypointId = keypoint["partId"] as Int
        val offsetY = offsets[heatmapY][heatmapX][keypointId]
        val offsetX = offsets[heatmapY][heatmapX][keypointId + numParts]
        val y = heatmapY * outputStride + offsetY
        val x = heatmapX * outputStride + offsetX
        return floatArrayOf(y, x)
    }

    fun withinNmsRadiusOfCorrespondingPoint(
        poses: List<Map<String, Any>>,
        squaredNmsRadius: Float,
        y: Float,
        x: Float,
        keypointId: Int
    ): Boolean {
        for (pose in poses) {
            val keypoints =
                pose["keypoints"] as Map<Int, Any>?
            val correspondingKeypoint =
                keypoints!![keypointId] as Map<String, Any>?
            val _x = correspondingKeypoint!!["x"] as Float * inputSize - x
            val _y = correspondingKeypoint["y"] as Float * inputSize - y
            val squaredDistance = _x * _x + _y * _y
            if (squaredDistance <= squaredNmsRadius) return true
        }
        return false
    }

    fun traverseToTargetKeypoint(
        edgeId: Int,
        sourceKeypoint: Map<String?, Any?>?,
        targetKeypointId: Int,
        scores: Array<Array<FloatArray>>,
        offsets: Array<Array<FloatArray>>,
        outputStride: Int,
        displacements: Array<Array<FloatArray>>
    ): MutableMap<String?, Any?> {
        val height = scores.size
        val width: Int = scores[0].size
        val numKeypoints: Int = scores[0][0].size
        val sourceKeypointY = sourceKeypoint!!["y"] as Float * inputSize
        val sourceKeypointX = sourceKeypoint["x"] as Float * inputSize
        val sourceKeypointIndices = getStridedIndexNearPoint(
            sourceKeypointY, sourceKeypointX,
            outputStride, height, width
        )
        val displacement =
            getDisplacement(edgeId, sourceKeypointIndices, displacements)
        val displacedPoint = floatArrayOf(
            sourceKeypointY + displacement[0],
            sourceKeypointX + displacement[1]
        )
        var targetKeypoint = displacedPoint
        val offsetRefineStep = 2
        for (i in 0 until offsetRefineStep) {
            val targetKeypointIndices = getStridedIndexNearPoint(
                targetKeypoint[0], targetKeypoint[1],
                outputStride, height, width
            )
            val targetKeypointY = targetKeypointIndices[0]
            val targetKeypointX = targetKeypointIndices[1]
            val offsetY =
                offsets[targetKeypointY][targetKeypointX][targetKeypointId]
            val offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints]
            targetKeypoint = floatArrayOf(
                targetKeypointY * outputStride + offsetY,
                targetKeypointX * outputStride + offsetX
            )
        }
        val targetKeypointIndices = getStridedIndexNearPoint(
            targetKeypoint[0], targetKeypoint[1],
            outputStride, height, width
        )
        val score = sigmoid(
            scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]
        )
        val keypoint: MutableMap<String?, Any?> =
            HashMap()
        keypoint["score"] = score
        keypoint["part"] = partNames[targetKeypointId]
        keypoint["y"] = targetKeypoint[0] / inputSize
        keypoint["x"] = targetKeypoint[1] / inputSize
        return keypoint
    }

    fun getStridedIndexNearPoint(
        _y: Float,
        _x: Float,
        outputStride: Int,
        height: Int,
        width: Int
    ): IntArray {
        val y_ = Math.round(_y / outputStride)
        val x_ = Math.round(_x / outputStride)
        val y = if (y_ < 0) 0 else if (y_ > height - 1) height - 1 else y_
        val x = if (x_ < 0) 0 else if (x_ > width - 1) width - 1 else x_
        return intArrayOf(y, x)
    }

    fun getDisplacement(
        edgeId: Int,
        keypoint: IntArray,
        displacements: Array<Array<FloatArray>>
    ): FloatArray {
        val numEdges: Int = displacements[0][0].size / 2
        val y = keypoint[0]
        val x = keypoint[1]
        return floatArrayOf(
            displacements[y][x][edgeId],
            displacements[y][x][edgeId + numEdges]
        )
    }

    fun getInstanceScore(
        keypoints: Map<Int, Map<String?, Any?>?>,
        numKeypoints: Int
    ): Float {
        var scores = 0f
        for ((_, value) in keypoints) scores += value!!["score"] as Float
        return scores / numKeypoints
    }

    private fun sigmoid(x: Float): Float {
        return (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
    }

    private fun softmax(vals: FloatArray) {
        var max = Float.NEGATIVE_INFINITY
        for (`val` in vals) {
            max = Math.max(max, `val`)
        }
        var sum = 0.0f
        for (i in vals.indices) {
            vals[i] = Math.exp(vals[i] - max.toDouble()).toFloat()
            sum += vals[i]
        }
        for (i in vals.indices) {
            vals[i] = vals[i] / sum
        }
    }


}


